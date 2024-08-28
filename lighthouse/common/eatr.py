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
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn

from lighthouse.common.eatr_transformer import Transformer, inverse_sigmoid, MLP, LinearLayer
from lighthouse.common.matcher import build_matcher, build_event_matcher
from lighthouse.common.position_encoding import build_position_encoding

from lighthouse.common.utils.span_utils import generalized_temporal_iou, generalized_temporal_iou_, span_cxw_to_xx
from lighthouse.common.misc import accuracy


class EaTR(nn.Module):
    """ This is the EaTR module that performs moment localization. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False, max_v_l=75, 
                 span_loss_type="l1", n_input_proj=2, query_dim=2, aud_dim=0):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         EaTR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        
        # model
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        self.n_input_proj = n_input_proj
        self.use_txt_pos = False

        hidden_dim = transformer.d_model
        self.num_dec_layers = transformer.dec_layers 

        # query
        self.num_queries = num_queries
        self.query_dim = query_dim

        # prediction
        self.max_v_l = max_v_l


        # loss
        self.span_loss_type = span_loss_type
        self.aux_loss = aux_loss

        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        # encoder
        # saliency prediction
        self.saliency_proj = nn.Linear(hidden_dim, 1)

        # decoder
        # span prediction
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2

        self.event_span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        nn.init.constant_(self.event_span_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.event_span_embed.layers[-1].bias.data, 0)

        self.moment_span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        nn.init.constant_(self.moment_span_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.moment_span_embed.layers[-1].bias.data, 0)
        # foreground classification
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground

        # iterative anchor update
        self.transformer.decoder.moment_span_embed = self.moment_span_embed
        self.transformer.decoder.event_span_embed = self.event_span_embed
 
    def generate_pseudo_event(self, src_vid, src_vid_mask):
        bsz, L_src, _ = src_vid.size()

        norm_vid = src_vid / (src_vid.norm(dim=2, keepdim=True)+1e-8)
        tsm = torch.bmm(norm_vid, norm_vid.transpose(1,2))
        mask = torch.tensor([[1., 1., 0., -1., -1.],
                             [1., 1., 0., -1., -1.],
                             [0., 0., 0., 0., 0.],
                             [-1., -1., 0., 1., 1.],
                             [-1., -1., 0., 1., 1.]], device=src_vid.device)
        mask_size = mask.size(0)
        mask = mask.view(1,mask_size,mask_size)
        pad_tsm = nn.ZeroPad2d(mask_size//2)(tsm)
        score = torch.diagonal(F.conv2d(pad_tsm.unsqueeze(1), mask.unsqueeze(1)).squeeze(1), dim1=1,dim2=2)  # [bsz,L_src]
        # average score as threshold
        tau = score.mean(1).unsqueeze(1).repeat(1,L_src)
        # fill the start, end indices with the max score
        L_vid = torch.count_nonzero(src_vid_mask,1)
        st_ed = torch.cat([torch.zeros_like(L_vid).unsqueeze(1), L_vid.unsqueeze(1)-1], dim=-1)
        score[torch.arange(score.size(0)).unsqueeze(1), st_ed] = 100
        # adjacent point removal and thresholding
        score_r = torch.roll(score,1,-1)
        score_l = torch.roll(score,-1,-1)
        bnds = torch.where((score_r<=score) & (score_l<=score) & (tau<=score), 1., 0.)

        bnd_indices = bnds.nonzero()
        temp = torch.roll(bnd_indices, 1, 0)
        center = (bnd_indices + temp) / 2
        width = bnd_indices - temp
        bnd_spans = torch.cat([center, width[:,1:]], dim=-1)
        pseudo_event_spans = [bnd_spans[bnd_spans[:,0] == i,:][:,1:]/L_vid[i] for i in range(bsz)]

        return pseudo_event_spans

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, src_aud=None, src_aud_mask=None):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)

        pseudo_event_spans = self.generate_pseudo_event(src_vid, src_vid_mask)  # comment the line for computational cost check

        src_vid = self.input_vid_proj(src_vid)
        
        src_txt = self.input_txt_proj(src_txt)  # (bsz, L_txt, d)
        src_txt_global = torch.max(src_txt, dim=1)[0]

        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)

        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt, src_txt_mask) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        # (#layers+1, bsz, #queries, d), (bsz, L_vid+L_txt, d), (#layers, bsz, #queries, query_dim)
        hs, memory, reference = self.transformer(src, ~mask, pos, src_vid, pos_vid, ~src_vid_mask.bool(), src_txt_global)
        
        reference_before_sigmoid = inverse_sigmoid(reference)
        event_tmp = self.event_span_embed(hs[0])
        event_outputs_coord = event_tmp.sigmoid()

        tmp = self.moment_span_embed(hs[-self.num_dec_layers:])
        tmp[..., :self.query_dim] += reference_before_sigmoid[-self.num_dec_layers:]
        outputs_coord = tmp.sigmoid()

        outputs_class = self.class_embed(hs[-self.num_dec_layers:])  # (#layers, batch_size, #queries, #classes)

        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}

        out['pseudo_event_spans'] = pseudo_event_spans  # comment the line for computational cost check
        out['pred_event_spans'] = event_outputs_coord

        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)

        out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze(-1)  # (bsz, L_vid)

        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            out['aux_outputs'] = [{'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        
        return out


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, span_loss_type, max_v_l,
                 saliency_margin=1, event_matcher=None):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.event_matcher = event_matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_event_spans(self, outputs, targets, indices):
        assert 'pred_event_spans' in outputs
        ## boundary span prediction
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_event_spans'][idx]
        tgt_spans = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_event_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
        loss_event_giou = 1 - torch.diag(generalized_temporal_iou_(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        return {
            'loss_event_span': loss_event_span.mean(),
            'loss_event_giou': loss_event_giou.mean(),
        }

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}
        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
            "event_spans": self.loss_event_spans,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        #self.epoch_i = epoch_i

        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux_outputs' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)
        moment_indices = self.matcher(outputs_without_aux, targets)
        event_indices = self.event_matcher(outputs['pred_event_spans'], outputs['pseudo_event_spans'])

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == 'event_spans':
                indices_in = event_indices
                targets_in = outputs['pseudo_event_spans']
            else:
                indices_in = moment_indices
                targets_in = targets
            losses.update(self.get_loss(loss, outputs, targets_in, indices_in))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ['saliency', 'event_spans']:
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/moment_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    query_dim = 2 # 1 for starting point only, 2 for both the starting point and the width
    transformer = Transformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        return_intermediate_dec=True,
        query_dim=query_dim,
        num_queries=args.num_queries,
        num_iteration=3, # Number of iterations for computing slot attention
    )

    model = EaTR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        aud_dim=args.a_feat_dim if "a_feat_dim" in args else 0,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        span_loss_type=args.span_loss_type,
        n_input_proj=args.n_input_proj,
        query_dim=query_dim,
    )

    matcher = build_matcher(args)
    event_matcher = build_event_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                "loss_giou": args.giou_loss_coef,
                "loss_label": args.label_loss_coef,
                "loss_saliency": args.lw_saliency,
                "loss_event_span": args.event_coef*args.span_loss_coef,
                "loss_event_giou": args.event_coef*args.giou_loss_coef,
                }

    if args.aux_loss:
        aux_weight_dict = {}
        weight_dict.update(aux_weight_dict)
        for i in range(args.dec_layers - 1):
            loss = ["loss_span", "loss_giou", "loss_label"]
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k in loss})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency', 'event_spans']

    criterion = SetCriterion(
        matcher=matcher, event_matcher=event_matcher, 
        weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, span_loss_type=args.span_loss_type, 
        max_v_l=args.max_v_l, saliency_margin=args.saliency_margin,
        )

    criterion.to(device)
    return model, criterion