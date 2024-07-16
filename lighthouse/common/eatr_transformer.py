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
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from lighthouse.common.attention import MultiheadAttention


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def gen_sineembed_for_position(pos_tensor, d_model):
    # pos_tensor [#queries, bsz, query_dim]
    scale = 2 * math.pi
    dim_t = torch.arange(d_model, dtype=torch.float32, device=pos_tensor.device)    # [d_model]
    dim_t = 10000 ** (2 * (dim_t // 2) / d_model)
    x_embed = pos_tensor[:, :, 0] * scale    # [#queries, bsz]
    pos_x = x_embed[:, :, None] / dim_t    # [#queries, bsz, d_model]
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)    # [#queries, bsz, d_model]

    w_embed = pos_tensor[:, :, 1] * scale    # [#queries, bsz]
    pos_w = w_embed[:, :, None] / dim_t    # [#queries, bsz, d_model]
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)    # [#queries, bsz, d_model]

    pos = torch.cat((pos_x, pos_w), dim=2)    # [#queries, bsz, 2*d_model]

    return pos


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu",
                 return_intermediate_dec=True, query_dim=2,
                 num_queries=10,
                 num_iteration=3):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        slot_atten = SlotAttention(num_iteration, num_queries, d_model)
        first_decoder_layer = TransformerDecoderFirstLayer(d_model, nhead, dim_feedforward,
                                                           dropout, activation)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        decoder_event_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(slot_atten, first_decoder_layer, decoder_layer,
                                          num_decoder_layers, decoder_norm, decoder_event_norm,
                                          return_intermediate=return_intermediate_dec, d_model=d_model,
                                          query_dim=query_dim)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, src_vid, pos_embed_vid, src_vid_mask, src_txt_global):

        src = src.permute(1, 0, 2)  # (L, bsz, d)
        pos_embed = pos_embed.permute(1, 0, 2)  # (L, bsz, d)

        src_txt_global = src_txt_global.unsqueeze(1).permute(1,0,2)  # (1, bsz, d)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # (L, bsz, d)
        hs, references = self.decoder(memory, memory_key_padding_mask=mask, 
                                      pos=pos_embed, 
                                      src_vid=src_vid,
                                      pos_vid=pos_embed_vid,
                                      src_vid_mask=src_vid_mask,                                      
                                      src_txt_global=src_txt_global)  # (#layers, bsz, #qeries, d)

        memory = memory.transpose(0, 1)  # (bsz, L, d)
        return hs, memory, references


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        intermediate = []

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

    
class TransformerDecoder(nn.Module):

    def __init__(self, slot_atten, first_decoder_layer, decoder_layer, num_layers, norm=None, event_norm=None, return_intermediate=False,
                d_model=512, query_dim=2):
        super().__init__()

        self.slot_atten = slot_atten
        self.layers = nn.ModuleList([])
        self.layers.append(first_decoder_layer)
        self.layers.extend(_get_clones(decoder_layer, num_layers-1))

        self.num_layers = num_layers
        self.norm = norm
        self.event_norm = event_norm
        self.return_intermediate = return_intermediate
        self.query_dim = query_dim
        self.d_model = d_model

        # DAB-DETR
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)  # position embedding dimension reduction
        self.query_scale = MLP(d_model, d_model, d_model, 2)  # center scaling
        self.ref_anchor_head = MLP(d_model, d_model, 1, 2)  # width modulation

        self.event_span_embed = None
        self.moment_span_embed = None

    def forward(self, memory, 
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                src_vid: Optional[Tensor] = None,
                pos_vid: Optional[Tensor] = None,
                src_vid_mask: Optional[Tensor] = None,
                src_txt_global: Optional[Tensor] = None):

        intermediate = []
        ref_points = []

        # Event reasoning
        src_slot = src_vid+pos_vid
        output = self.slot_atten(src_slot, src_vid_mask)  # [bsz, #queries, d_model]
        output = output.permute(1,0,2)  # [#queries, bsz, d_model]

        if self.event_span_embed:
            tmp = self.event_span_embed(output)
            new_reference_points = tmp[..., :self.query_dim].sigmoid()
            ref_points.append(new_reference_points)
            reference_points = new_reference_points.detach()

        if self.return_intermediate:
            intermediate.append(self.event_norm(output))

        # Moment reasoning
        for layer_id, layer in enumerate(self.layers):
            ref_pt = reference_points[..., :self.query_dim]  # [#queries, bsz, 2] (xw)

            query_sine_embed = gen_sineembed_for_position(ref_pt, self.d_model)  # [#queries, bsz, 2*d_model] - (:d_model)은 center, (d_model:)은 width (xw)
            query_pos = self.ref_point_head(query_sine_embed)  # [#queries, bsz, d_model] (xw)

            # Conditional-DETR
            pos_transformation = self.query_scale(output)
            query_sine_embed = query_sine_embed[...,:self.d_model] * pos_transformation  # [#queries, bsz, d_model] (x)
            # modulated w attentions
            refW_cond = self.ref_anchor_head(output).sigmoid()  # [#queries, bsz, 1] (w)
            query_sine_embed *= (refW_cond[..., 0] / ref_pt[..., 1]).unsqueeze(-1)  # (x *= w/w)

            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, 
                           query_sine_embed=query_sine_embed,
                           src_txt_global=src_txt_global)

            if self.moment_span_embed:
                tmp = self.moment_span_embed(output)
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.moment_span_embed:
                return [
                    torch.stack(intermediate).transpose(1,2),
                    torch.stack(ref_points).transpose(1,2)
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1,2),
                    reference_points.unsqueeze(0).transpose(1,2)
                ]

        return output.unsqueeze(0)



class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_iterations, num_slots, d_model,
                epsilon=1e-8):
        """Builds the Slot Attention module.
        Args:
            num_iterations: Number of iterations.
            num_slots: Number of slots.
            d_model: Hidden layer size of MLP.
            epsilon: Offset for attention coefficients before normalization.
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.d_model = d_model
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(d_model)
        self.norm_slots = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

        self.slots = nn.Parameter(torch.randn(num_slots, d_model))
        nn.init.xavier_normal_(self.slots)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(d_model, d_model)
        self.project_k = nn.Linear(d_model, d_model)
        self.project_v = nn.Linear(d_model, d_model)

        self.attn_holder = nn.Identity()

        # Slot update functions.
        self.mlp = MLP(d_model, d_model, d_model, 2)

    def forward(self, inputs, mask):
        b = inputs.shape[0]  # [bsz, n_inputs, d_model]

        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input
        k = self.project_k(inputs)  # [bsz, n_inputs, d_model]
        v = self.project_v(inputs)   # [bsz, n_inputs, d_model]

        slots = self.slots.repeat(b,1,1)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)   # [bsz, num_slots, d_model]
            scale = self.d_model ** -0.5    # Normalization

            dots = torch.einsum('bid,bjd->bij', q, k) * scale  # [bsz, num_slots, n_inputs]

            max_neg_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(mask.unsqueeze(1), max_neg_value)

            attn = dots.softmax(dim=1)
            attn = self.attn_holder(attn)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)  # softmax over slots
            updates = torch.einsum('bjd,bij->bid', v, attn)  # [bsz, num_slots, d_model].

            # Slot update.
            slots = slots_prev + updates
            slots = slots + self.mlp(self.norm_mlp(slots))

        return self.norm_out(slots)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()

        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     src_txt_global: Optional[Tensor] = None,):
                     
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # [#queries, batch_size, d_model]
        q_content = self.sa_qcontent_proj(tgt)
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # [#queries, batch_size, d_model]
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        q = q_content
        k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                               key=k,
                               value=v, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoderFirstLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()

        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Gated Fusion
        self.gate_cross_attn = nn.MultiheadAttention(d_model, 1, dropout=0.1)
        self.gate_self_attn = nn.MultiheadAttention(d_model, 1, dropout=0.1)
        self.gate_dropout = nn.Dropout(dropout)
        self.gate_norm = nn.LayerNorm(d_model)
        self.gate_linear = nn.Linear(d_model, d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     src_txt_global: Optional[Tensor] = None,):
        
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # [#queries, batch_size, d_model]
        q_content = self.sa_qcontent_proj(tgt)
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)   # C'

        # ========== Begin of Gated Fusion =============
        
        tgt2 = self.gate_cross_attn(query=tgt,
                                    key=src_txt_global,
                                    value=src_txt_global)[0]  # \hat{C}
        gate = (tgt*tgt2).sigmoid()
        tgt2 = tgt+tgt2
        tgt2 = self.gate_self_attn(query=tgt2, 
                                   key=tgt2, 
                                   value=tgt2)[0]
        tgt = self.gate_dropout(self.activation(self.gate_linear(gate*(tgt2)))) + tgt

        # ========== End of Gated Fusion =============
        tgt = self.gate_norm(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # [#queries, batch_size, d_model]
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)
        q_pos = self.ca_qpos_proj(query_pos)

        q = q_content + q_pos
        k = k_content + k_pos

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                                key=k,
                                value=v, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")