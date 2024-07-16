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

MIT License

Copyright (c) 2022 Anonymous

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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# VCTC
class CTC_Loss(nn.Module):
    def __init__(self, temperature=0.07):
        super(CTC_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, vid_feat, txt_feat, pos_mask, src_vid_mask=None, src_txt_mask=None):
        # vid_feat: (bs, t, d)
        # txt_feat: (bs, n, d)
        # pos_mask: (bs, t)
        # src_vid_mask: (bs, t) or None
        # src_txt_mask: (bs, n) or None
        bs = vid_feat.size(0)
        t = vid_feat.size(1)
        n = txt_feat.size(1)
        d = vid_feat.size(2)
        # normalize the feature vectors
        vid_feat = F.normalize(vid_feat, dim=2) # (bs, t, d)
        txt_feat = F.normalize(txt_feat, dim=2) # (bs, n, d)
        # compute the global text feature by mean pooling
        if src_txt_mask is not None:
            src_txt_mask = src_txt_mask.unsqueeze(-1) # (bs, n, 1)
            txt_feat = txt_feat * src_txt_mask # (bs, n, d)
            txt_global = torch.sum(txt_feat, dim=1) / torch.sum(src_txt_mask, dim=1) # (bs, d)
        else:
            txt_global = torch.mean(txt_feat, dim=1) # (bs, d)
        # compute the similarity matrix
        sim_mat = torch.bmm(vid_feat, txt_global.unsqueeze(-1)).squeeze(-1) # (bs, t)
        # apply the video mask if given
        if src_vid_mask is not None:
            sim_mat = sim_mat * src_vid_mask # (bs, t)
        # compute the logits and labels
        logits = sim_mat / self.temperature # (bs, t)
        labels = pos_mask.long() # (bs, t)
        # compute the binary cross entropy loss with logits
        loss = F.binary_cross_entropy_with_logits(logits, labels.float()) # scalar
        # return the loss
        return loss


class VTCLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(VTCLoss, self).__init__()
        self.temperature = temperature

    def forward(self, src_txt, src_vid):
        # src_txt: (bs, h_dim)
        # src_vid: (bs, h_dim)
        bs = src_txt.size(0)
        h_dim = src_txt.size(1)
        # normalize the feature vectors
        src_txt = F.normalize(src_txt, dim=1)
        src_vid = F.normalize(src_vid, dim=1)
        # compute the similarity matrix
        sim_mat = torch.mm(src_txt, src_vid.t()) # (bs, bs)
        # create the positive and negative masks
        pos_mask = torch.eye(bs).bool().to(sim_mat.device) # (bs, bs)
        neg_mask = ~pos_mask # (bs, bs)
        # compute the logits and labels
        logits = sim_mat / self.temperature # (bs, bs)
        labels = torch.arange(bs).to(sim_mat.device) # (bs,)
        # compute the cross entropy loss for text-to-video and video-to-text
        loss_t2v = F.cross_entropy(logits, labels) # scalar
        loss_v2t = F.cross_entropy(logits.t(), labels) # scalar
        # return the average loss
        return (loss_t2v + loss_v2t) / 2