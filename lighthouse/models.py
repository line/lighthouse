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

import torch
import torch.nn.functional as F

from lighthouse.common.qd_detr import build_model as build_model_qd_detr
from lighthouse.common.moment_detr import build_model as build_model_moment_detr
from lighthouse.common.cg_detr import build_model as build_model_cg_detr
from lighthouse.common.eatr import build_model as build_model_eatr
from lighthouse.common.uvcom import build_model as build_model_uvcom
from lighthouse.common.tr_detr import build_model as build_model_tr_detr
from lighthouse.common.taskweave import build_model as build_model_task_weave

from lighthouse.common.utils.span_utils import span_cxw_to_xx
from lighthouse.feature_extractor import VideoFeatureExtractor


class BasePredictor:
    def __init__(self, model_name, ckpt_path, device, feature_name, slowfast_path=None, pann_path=None):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        args = ckpt["opt"]
        self.clip_len = args.clip_length
        self.device = device
        args.device = device # for CPU users
        self.feature_extractor = VideoFeatureExtractor(
            framerate=1/self.clip_len, size=224, centercrop=(feature_name != 'resnet_glove'),
            feature_name=feature_name, device=device, slowfast_path=slowfast_path, pann_path=pann_path,
        )
        if model_name == 'moment_detr':
            self.model, _ = build_model_moment_detr(args)
        elif model_name == 'qd_detr':
            self.model, _ = build_model_qd_detr(args)
        elif model_name == 'eatr':
            self.model, _ = build_model_eatr(args)
        elif model_name == 'cg_detr':
            self.model, _ = build_model_cg_detr(args)
        elif model_name == 'tr_detr':
            self.model, _ = build_model_tr_detr(args)
        elif model_name == 'uvcom':
            self.model, _ = build_model_uvcom(args)
        elif model_name == 'taskweave':
            self.model, _ = build_model_task_weave(args)
        else:
            raise NotImplementedError
        
        self.feature_name = feature_name
        self.model_name = model_name
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()
        self.video_feats = None
        self.video_mask = None
        self.video_path = None

    @torch.no_grad()
    def encode_video_audio(self, video_path):
        # construct model inputs
        video_feats = self.feature_extractor.encode_video(video_path)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(video_feats)
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)
        video_feats = torch.cat([video_feats, tef], dim=1)

        audio_feats = self.feature_extractor.encode_audio(video_path)

        assert n_frames <= 75, "The positional embedding only support video up to 150 secs (i.e., 75 2-sec clips) in length"
        video_feats = video_feats.unsqueeze(0)
        video_mask = torch.ones(1, n_frames).to(self.device)
        self.video_feats = video_feats
        self.video_mask = video_mask
        self.video_path = video_path

    @torch.no_grad()
    def encode_video(self, video_path):
        # construct model inputs
        video_feats = self.feature_extractor.encode_video(video_path)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(video_feats)
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)
        video_feats = torch.cat([video_feats, tef], dim=1)
        assert n_frames <= 75, "The positional embedding only support video up to 150 secs (i.e., 75 2-sec clips) in length"
        video_feats = video_feats.unsqueeze(0)
        video_mask = torch.ones(1, n_frames).to(self.device)
        self.video_feats = video_feats
        self.video_mask = video_mask
        self.video_path = video_path

    @torch.no_grad()
    def predict(self, query):
        if self.video_feats is None or self.video_mask is None or self.video_path is None:
            # raise ValueError('Video features are not encoded. First run .encode_video() before predict().')
            return None

        query_feats, query_mask = self.feature_extractor.encode_text(query)  # #text * (L, d) -> CLIP or GloVe
        if self.feature_name != "resnet_glove":
            query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)

        if self.model_name == 'cg_detr':
            model_inputs = dict(
                src_vid=self.video_feats,
                src_vid_mask=self.video_mask,
                src_txt=query_feats,
                src_txt_mask=query_mask,
                vid=None, qid=None
            )
        else:
            model_inputs = dict(
                src_vid=self.video_feats,
                src_vid_mask=self.video_mask,
                src_txt=query_feats,
                src_txt_mask=query_mask
            )
            
        # decode outputs
        if self.model_name == 'taskweave':
            model_inputs["epoch_i"] = None
            outputs, _ = self.model(**model_inputs)
        else:
            outputs = self.model(**model_inputs)
        prob = F.softmax(outputs["pred_logits"], -1).squeeze(0).cpu()
        scores = prob[:,0]
        pred_spans = outputs["pred_spans"].squeeze(0).cpu()
        saliency_scores = outputs["saliency_scores"][model_inputs["src_vid_mask"] == 1].cpu().tolist()

        # compose prediction
        video_duration = self.video_feats.shape[1] * self.clip_len
        pred_spans = span_cxw_to_xx(pred_spans) * video_duration
        pred_spans = torch.clamp(pred_spans, min=0, max=video_duration)
        cur_ranked_preds = torch.cat([pred_spans, scores[:, None]], dim=1).tolist()
        cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
        cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
        prediction = {
            "query": query,
            "pred_relevant_windows": cur_ranked_preds,
            "pred_saliency_scores": saliency_scores,
        }
        return prediction


class MomentDETRPredictor(BasePredictor):
    def __init__(self, ckpt_path, device, feature_name, slowfast_path):
        super().__init__('moment_detr', ckpt_path, device, feature_name, slowfast_path)


class QDDETRPredictor(BasePredictor):
    def __init__(self, ckpt_path, device, feature_name, slowfast_path):
        super().__init__('qd_detr', ckpt_path, device, feature_name, slowfast_path)


class EaTRPredictor(BasePredictor):
    def __init__(self, ckpt_path, device, feature_name, slowfast_path):
        super().__init__('eatr', ckpt_path, device, feature_name, slowfast_path)


class CGDETRPredictor(BasePredictor):
    def __init__(self, ckpt_path, device, feature_name, slowfast_path):
        super().__init__('cg_detr', ckpt_path, device, feature_name, slowfast_path)


class TRDETRPredictor(BasePredictor):
    def __init__(self, ckpt_path, device, feature_name, slowfast_path):
        super().__init__('tr_detr', ckpt_path, device, feature_name, slowfast_path)


class UVCOMPredictor(BasePredictor):
    def __init__(self, ckpt_path, device, feature_name, slowfast_path):
        super().__init__('uvcom', ckpt_path, device, feature_name, slowfast_path)


class TaskWeavePredictor(BasePredictor):
    def __init__(self, ckpt_path, device, feature_name, slowfast_path):
        super().__init__('taskweave', ckpt_path, device, feature_name, slowfast_path)