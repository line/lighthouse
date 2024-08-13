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

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from lighthouse.common.utils.span_utils import generalized_temporal_iou, generalized_temporal_iou_, span_cxw_to_xx


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,  cost_class: float = 1, cost_span: float = 1, cost_giou: float = 1,
                 span_loss_type: str = "l1", max_v_l: int = 75):
        """Creates the matcher

        Params:
            cost_span: This is the relative weight of the L1 error of the span coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the spans in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.foreground_label = 0
        assert cost_class != 0 or cost_span != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_spans": Tensor of dim [batch_size, num_queries, 2] with the predicted span coordinates,
                    in normalized (cx, w) format
                 ""pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "spans": Tensor of dim [num_target_spans, 2] containing the target span coordinates. The spans are
                    in normalized (cx, w) format

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_spans)
        """
        bs, num_queries = outputs["pred_spans"].shape[:2]
        targets = targets["span_labels"]

        # Also concat the target labels and spans
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        tgt_spans = torch.cat([v["spans"] for v in targets])  # [num_target_spans in batch, 2]
        tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)   # [total #spans in the batch]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]  # [batch_size * num_queries, total #spans in the batch]

        if self.span_loss_type == "l1":
            # We flatten to compute the cost matrices in a batch
            out_spans = outputs["pred_spans"].flatten(0, 1)  # [batch_size * num_queries, 2]

            # Compute the L1 cost between spans
            cost_span = torch.cdist(out_spans, tgt_spans, p=1)  # [batch_size * num_queries, total #spans in the batch]

            # Compute the giou cost between spans
            # [batch_size * num_queries, total #spans in the batch]
            cost_giou = - generalized_temporal_iou(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))
        else:
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, max_v_l * 2)
            pred_spans = pred_spans.view(bs * num_queries, 2, self.max_v_l).softmax(-1)  # (bsz * #queries, 2, max_v_l)
            cost_span = - pred_spans[:, 0][:, tgt_spans[:, 0]] - \
                pred_spans[:, 1][:, tgt_spans[:, 1]]  # (bsz * #queries, #spans)
            # pred_spans = pred_spans.repeat(1, n_spans, 1, 1).flatten(0, 1)  # (bsz * #queries * #spans, max_v_l, 2)
            # tgt_spans = tgt_spans.view(1, n_spans, 2).repeat(bs * num_queries, 1, 1).flatten(0, 1)  # (bsz * #queries * #spans, 2)
            # cost_span = pred_spans[tgt_spans]
            # cost_span = cost_span.view(bs * num_queries, n_spans)

            # giou
            cost_giou = 0

        # Final cost matrix
        C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["spans"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianEventMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cost_span: float = 1, cost_giou: float = 1,
                 span_loss_type: str = "l1", max_v_l: int = 75):
        """Creates the matcher

        Params:
            cost_span: This is the relative weight of the L1 error of the span coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the spans in the matching cost
        """
        super().__init__()
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.foreground_label = 0
        assert cost_span != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_spans": Tensor of dim [batch_size, num_queries, 2] with the predicted span coordinates,
                    in normalized (cx, w) format
                 ""pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "spans": Tensor of dim [num_target_spans, 2] containing the target span coordinates. The spans are
                    in normalized (cx, w) format

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_spans)
        """
        bs, num_queries = outputs.shape[:2]

        # Also concat the target labels and spans
        tgt_spans = torch.cat([v for v in targets])  # [num_target_spans in batch, 2]

        # We flatten to compute the cost matrices in a batch
        out_spans = outputs.flatten(0, 1)  # [batch_size * num_queries, 2]

        # Compute the L1 cost between spans
        cost_span = torch.cdist(out_spans, tgt_spans, p=1)  # [batch_size * num_queries, total #spans in the batch]

        # Compute the giou cost between spans
        # [batch_size * num_queries, total #spans in the batch]
        cost_giou = - generalized_temporal_iou_(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))
        
        # Final cost matrix
        C = self.cost_span * cost_span + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(
        cost_span=args.set_cost_span, cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class, span_loss_type=args.span_loss_type, max_v_l=args.max_v_l
    )


def build_event_matcher(args):
    return HungarianEventMatcher(
        cost_span=args.set_cost_span, cost_giou=args.set_cost_giou,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l
    )