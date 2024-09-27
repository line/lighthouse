"""
Copyright $today.year LY Corporation

LY Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

Moment-DETR (https://github.com/jayleicn/moment_detr)
Copyright (c) 2021 Jie Lei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import pprint

from tqdm import tqdm, trange
import numpy as np
import os
from collections import OrderedDict, defaultdict
from easydict import EasyDict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lighthouse.common.utils.basic_utils import AverageMeter
from lighthouse.common.utils.span_utils import span_cxw_to_xx

from training.config import BaseOptions

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from training.dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from training.cg_detr_dataset import CGDETR_StartEndDataset, cg_detr_start_end_collate, cg_detr_prepare_batch_inputs

from training.postprocessing import PostProcessorDETR
from standalone_eval.eval import eval_submission

from lighthouse.common.utils.basic_utils import save_jsonl, save_json
from lighthouse.common.qd_detr import build_model as build_model_qd_detr
from lighthouse.common.moment_detr import build_model as build_model_moment_detr
from lighthouse.common.cg_detr import build_model as build_model_cg_detr
from lighthouse.common.eatr import build_model as build_model_eatr
from lighthouse.common.tr_detr import build_model as build_model_tr_detr
from lighthouse.common.uvcom import build_model as build_model_uvcom
from lighthouse.common.taskweave import build_model as build_model_task_weave

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)

    if opt.eval_split_name in ["val"]:
        metrics = eval_submission(submission, gt_data)
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path, ]

    return metrics, latest_file_paths


# for HL
@torch.no_grad()
def compute_hl_results(epoch_i, model, eval_loader, opt, criterion=None):
    batch_input_fn = cg_detr_prepare_batch_inputs  if opt.model_name == 'cg_detr' else prepare_batch_inputs
    loss_meters = defaultdict(AverageMeter)

    video_ap_collected = []
    topk = 5 # top-5 map

    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]
        model_inputs, targets = batch_input_fn(batch[1], opt.device)

        if opt.model_name == 'taskweave':
            model_inputs['epoch_i'] = epoch_i
            outputs, _ = model(**model_inputs)
        else:
            outputs = model(**model_inputs)

        preds = outputs['saliency_scores']
        for meta, pred in zip(query_meta, preds):
            label = meta['label'] # raw label
            video_ap = []
            # Follow the UMT code "https://github.com/TencentARC/UMT/blob/main/datasets/tvsum.py"
            if opt.dset_name == 'tvsum':
                for i in range(20):
                    pred = pred.cpu()
                    cur_pred = pred[:len(label)]
                    inds = torch.argsort(cur_pred, descending=True, dim=-1)

                    # video_id = self.get_video_id(idx)
                    cur_label = torch.Tensor(label)[:, i]
                    cur_label = torch.where(cur_label > cur_label.median(), 1.0, .0)

                    cur_label = cur_label[inds].tolist()[:topk]

                    # if (num_gt := sum(cur_label)) == 0:
                    num_gt = sum(cur_label)
                    if num_gt == 0:
                        video_ap.append(0)
                        continue

                    hits = ap = rec = 0
                    prc = 1

                    for j, gt in enumerate(cur_label):
                        hits += gt

                        _rec = hits / num_gt
                        _prc = hits / (j + 1)

                        ap += (_rec - rec) * (prc + _prc) / 2
                        rec, prc = _rec, _prc

                    video_ap.append(ap)
            
            elif opt.dset_name == 'youtube_highlight':
                cur_pred = pred[:len(label)].cpu()
                inds = torch.argsort(cur_pred, descending=True, dim=-1)
                cur_label = torch.Tensor(label).squeeze()[inds].tolist()
                num_gt = sum(cur_label)
                if num_gt == 0:
                    video_ap.append(0)
                    continue

                hits = ap = rec = 0
                prc = 1

                for j, gt in enumerate(cur_label):
                    hits += gt

                    _rec = hits / num_gt
                    _prc = hits / (j + 1)

                    ap += (_rec - rec) * (prc + _prc) / 2
                    rec, prc = _rec, _prc
                
                video_ap.append(float(ap))

            else:
                raise NotImplementedError

            video_ap_collected.append(video_ap)  

    mean_ap = np.mean(video_ap_collected)
    submmission = dict(mAP=round(mean_ap, 5))
    
    return submmission, loss_meters


@torch.no_grad()
def compute_mr_results(epoch_i, model, eval_loader, opt, criterion=None):
    batch_input_fn = cg_detr_prepare_batch_inputs if opt.model_name == 'cg_detr' else prepare_batch_inputs
    loss_meters = defaultdict(AverageMeter)

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]
        model_inputs, targets = batch_input_fn(batch[1], opt.device)

        if opt.model_name == 'taskweave':
            model_inputs['epoch_i'] = epoch_i
            outputs, _ = model(**model_inputs)
        else:
            outputs = model(**model_inputs)

        # saliency scores
        _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
        saliency_scores = []
        valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
        for j in range(len(valid_vid_lengths)):
            valid_length = int(valid_vid_lengths[j])
            saliency_scores.append(_saliency_scores[j, :valid_length].tolist())

        # compose predictions
        pred_spans = outputs["pred_spans"].cpu()  # (bsz, #queries, 2)
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #queries, #classes=2)
        scores = prob[..., 0].cpu()  # * (batch_size, #queries)  foreground label is 0, we directly take it

        for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans, scores)):            
            spans = span_cxw_to_xx(spans) * meta["duration"]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]

            if opt.dset_name in ['qvhighlight', 'qvhighlight_pretrain']:
                cur_query_pred = dict(
                    qid=meta["qid"],
                    query=meta["query"],
                    vid=meta["vid"],
                    pred_relevant_windows=cur_ranked_preds,
                    pred_saliency_scores=saliency_scores[idx]
                )
            else:
                # anet, charades
                cur_query_pred = dict(
                    qid=meta["qid"],
                    query=meta["query"],
                    vid=meta["vid"],
                    pred_relevant_windows=cur_ranked_preds,
                )

            mr_res.append(cur_query_pred)

        if criterion:
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict["loss_overall"] = float(losses)
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

    if opt.dset_name in ['qvhighlight', 'qvhighlight_pretrain']:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length, min_ts_val=0, max_ts_val=150,
            min_w_l=2, max_w_l=150, move_window_method="left",
            process_func_names=("clip_ts", "round_multiple")
        )
    elif opt.dset_name in ['charades', 'clotho-moment', 'unav100-subset', 'tut2017']:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length, min_ts_val=0, max_ts_val=150,
            min_w_l=2, max_w_l=60, move_window_method="left",
            process_func_names=("clip_ts", "round_multiple")
        )
    elif opt.dset_name in ['tacos', 'activitynet', 'youtube_highlight']:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length, min_ts_val=0, max_ts_val=50000,
            min_w_l=0, max_w_l=50000, move_window_method="left",
            process_func_names=(["round_multiple"])
        )
    else:
        raise NotImplementedError

    mr_res = post_processor(mr_res)
    return mr_res, loss_meters


def get_eval_res(epoch_i, model, eval_loader, opt, criterion):
    """compute and save query and video proposal embeddings"""
    eval_res, eval_loss_meters = compute_mr_results(epoch_i, model, eval_loader, opt, criterion)
    return eval_res, eval_loss_meters


def eval_epoch(epoch_i, model, eval_dataset, opt, save_submission_filename, criterion=None):
    collate_fn = cg_detr_start_end_collate if opt.model_name == 'cg_detr' else start_end_collate
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None:
        criterion.eval()

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=collate_fn,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
    )

    if opt.dset_name == 'tvsum' or opt.dset_name == 'youtube_highlight':
        metrics, eval_loss_meters = compute_hl_results(epoch_i, model, eval_loader, opt, criterion)
        # to match original save format
        submission = [{ "brief" : metrics }]
        save_metrics_path = os.path.join(opt.results_dir, save_submission_filename.replace('.jsonl', '_metrics.jsonl'))
        save_jsonl(submission, save_metrics_path)
        return submission[0], eval_loss_meters, [save_metrics_path]
    else:
        submission, eval_loss_meters = get_eval_res(epoch_i, model, eval_loader, opt, criterion)        
        metrics, latest_file_paths = eval_epoch_post_processing(
            submission, opt, eval_dataset.data, save_submission_filename)
        return metrics, eval_loss_meters, latest_file_paths

def build_model(opt):
    if opt.model_name == 'qd_detr':
        model, criterion = build_model_qd_detr(opt)
    elif opt.model_name == 'moment_detr':
        model, criterion = build_model_moment_detr(opt)
    elif opt.model_name == 'cg_detr':
        model, criterion = build_model_cg_detr(opt)
    elif opt.model_name == 'eatr':
        model, criterion = build_model_eatr(opt)
    elif opt.model_name == 'tr_detr':
        model, criterion = build_model_tr_detr(opt)
    elif opt.model_name == 'uvcom':
        model, criterion = build_model_uvcom(opt)
    elif opt.model_name == 'taskweave':
        model, criterion = build_model_task_weave(opt)
    else:
        raise NotImplementedError
    
    return model, criterion

def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    model, criterion = build_model(opt)

    if opt.device == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)

    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)

    return model, criterion, optimizer, lr_scheduler


def start_inference(opt, domain=None):
    logger.info("Setup config, data and model...")

    cudnn.benchmark = True
    cudnn.deterministic = False
    load_labels = opt.eval_split_name == 'val'
    epoch_i = None # for TaskWeave.
    
    # dataset & data loader
    dataset_config = EasyDict(
        dset_name=opt.dset_name,
        domain=domain,
        data_path=opt.eval_path,
        ctx_mode=opt.ctx_mode,
        v_feat_dirs=opt.v_feat_dirs,
        a_feat_dirs=opt.a_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        v_feat_types=opt.v_feat_types,
        a_feat_types=opt.a_feat_types,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=load_labels,
    )
    
    eval_dataset = CGDETR_StartEndDataset(**dataset_config) if opt.model_name == 'cg_detr' else StartEndDataset(**dataset_config)
    model, criterion, _, _ = setup_model(opt)
    checkpoint = torch.load(opt.model_path)
    model.load_state_dict(checkpoint["model"])
    logger.info("Model checkpoint: {}".format(opt.model_path))
    if not load_labels:
        criterion = None

    save_submission_filename = "hl_{}_submission.jsonl".format(opt.eval_split_name)

    logger.info("Starting inference...")
    with torch.no_grad():
        metrics, eval_loss_meters, latest_file_paths = \
            eval_epoch(epoch_i, model, eval_dataset, opt, save_submission_filename, criterion)

    if opt.eval_split_name == 'val':
        logger.info("metrics_no_nms {}".format(pprint.pformat(metrics["brief"], indent=4)))


def check_valid_combination(dataset, feature):
    if feature == 'i3d_clip':
        return dataset == 'tvsum'
    
    if feature == 'clip_slowfast_pann':
        return dataset == 'qvhighlight' or dataset == 'qvhighlight_pretrain'
    
    if dataset == 'youtube_highlight':
        # Due to unavailable access to the original videos, we publish only CLIP and CLIP+Slowfast for YouTube Highlight.
        return dataset != 'resnet_glove'
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True, 
                        choices=['moment_detr', 'qd_detr', 'eatr', 'cg_detr', 'uvcom', 'tr_detr', 'taskweave_hd2mr', 'taskweave_mr2hd'],
                        help='model name. select from [moment_detr, qd_detr, eatr, cg_detr, uvcom, tr_detr, taskweave_hd2mr, taskweave_mr2hd]')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        choices=['activitynet', 'charades', 'qvhighlight', 'qvhighlight_pretrain', 'tacos', 'tvsum', 'youtube_highlight', 'clotho-moment', 'unav100-subset', 'tut2017'],
                        help='dataset name. select from [activitynet, charades, qvhighlight, qvhighlight_pretrain, tacos, tvsum, youtube_highlight, clotho-moment, unav100-subset, tut2017]')
    parser.add_argument('--feature', '-f', type=str, required=True,
                        choices=['resnet_glove', 'clip', 'clip_slowfast', 'clip_slowfast_pann', 'i3d_clip', 'clap'],
                        help='feature name. select from [resnet_glove, clip, clip_slowfast, clip_slowfast_pann, i3d_clip, clap].'
                             'NOTE: i3d_clip and clip_slowfast_pann are only for TVSum and QVHighlight, respectively')
    parser.add_argument('--model_path', type=str, required=True, help='saved model path')
    parser.add_argument('--split', type=str, required=True, choices=['val', 'test'], help='val or test')
    parser.add_argument('--eval_path', type=str, required=True, help='evaluation data')
    args = parser.parse_args()

    is_valid = check_valid_combination(args.dataset, args.feature)

    if is_valid:
        option_manager = BaseOptions(args.model, args.dataset, args.feature)
        option_manager.parse()
        opt = option_manager.option
        os.makedirs(opt.results_dir, exist_ok=True)

        opt.model_path = args.model_path
        opt.eval_split_name = args.split
        opt.eval_path = args.eval_path
        
        if 'domains' in opt:
            for domain in opt.domains:
                opt.results_dir = os.path.join(opt.results_dir, domain)
                start_inference(opt, domain=domain)
        else:
            start_inference(opt)
    
    else:
        raise ValueError('The combination of dataset and feature is invalid: dataset={}, feature={}'.format(args.dataset, args.feature))
