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

from lighthouse.common.qd_detr import QDDETR
from lighthouse.common.moment_detr import MomentDETR
from lighthouse.common.cg_detr import CGDETR
from lighthouse.common.eatr import EaTR
from lighthouse.common.uvcom import UVCOM
from lighthouse.common.tr_detr import TRDETR
from lighthouse.common.taskweave import TaskWeave

from lighthouse.common.utils.span_utils import span_cxw_to_xx
from lighthouse.feature_extractor import VideoFeatureExtractor

from typing import Optional, Union, Mapping, Any, Dict, List, Tuple

class BasePredictor:
    def __init__(
        self,
        model_name: str,
        ckpt_path: str,
        device: str,
        feature_name: str,
        slowfast_path: Optional[str] = None,
        pann_path: Optional[str] = None) -> None:
        
        ckpt = torch.load(ckpt_path, map_location='cpu')
        args = ckpt['opt']
        self.clip_len: float = args.clip_length
        self.device: str = device
        args.device = device # for CPU users
        self.feature_extractor = VideoFeatureExtractor(
            framerate=1/self.clip_len, size=224, centercrop=(feature_name != 'resnet_glove'),
            feature_name=feature_name, device=device, slowfast_path=slowfast_path, pann_path=pann_path,
        )
        self.model: Union[MomentDETR, QDDETR, EaTR, UVCOM, TaskWeave, CGDETR, TRDETR] = self.initialize_model(args, model_name)
        self.load_weights(ckpt['model'])
        self.feature_name: str = feature_name
        self.model_name: str = model_name
        self.video_feats: Optional[torch.Tensor] = None
        self.video_mask: Optional[torch.Tensor] = None
        self.video_path: Optional[str] = None
        self.audio_feats: Optional[torch.Tensor] = None

    def initialize_model(
        self,
        args: Dict[str, Union[str, float, int, List[str]]],
        model_name: str) -> Union[MomentDETR, QDDETR, EaTR, UVCOM, TaskWeave, CGDETR, TRDETR]:
        
        model_builders = {
            'moment_detr': build_model_moment_detr,
            'qd_detr': build_model_qd_detr,
            'eatr': build_model_eatr,
            'cg_detr': build_model_cg_detr,
            'tr_detr': build_model_tr_detr,
            'uvcom': build_model_uvcom,
            'taskweave': build_model_task_weave
        }

        if model_name in model_builders:
            model, _ = model_builders[model_name](args)
            return model
        else:
            raise NotImplementedError(f'The {model_name} is not implemented. Choose from'
                                      '[moment_detr, qd_detr, eatr, cg_detr, tr_detr, uvcom, taskweave]')

    def load_weights(
        self, 
        model_weight: Mapping[str, Any]) -> None:
        
        self.model.load_state_dict(model_weight)
        self.model.to(self.device)
        self.model.eval()

    def normalize_and_concat_with_timestamps(
        self,
        video_feats: torch.Tensor) -> torch.Tensor:
        normalized_video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(normalized_video_feats)
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)
        timestamped_video_feats = torch.cat([video_feats, tef], dim=1)
        return timestamped_video_feats

    @torch.no_grad()
    def encode_video(
        self,
        video_path: str) -> None:
        video_feats: torch.Tensor = self.feature_extractor.encode_video(video_path)
        timestamed_video_feats: torch.Tensor = self.normalize_and_concat_with_timestamps(video_feats)
        n_frames: int = len(timestamed_video_feats)

        if n_frames > 75:
            raise ValueError('The positional embedding only support video up to 150 secs (i.e., 75 2-sec clips) in length')
        
        timestamed_video_feats = timestamed_video_feats.unsqueeze(0)
        video_mask: torch.Tensor = torch.ones(1, n_frames).to(self.device)
        self.video_feats = timestamed_video_feats
        self.video_mask = video_mask
        self.video_path = video_path
        if self.feature_name == 'clip_slowfast_pann':
            self.audio_feats = self.encode_audio(video_path)
    
    def encode_audio(
        self,
        video_path: str) -> torch.Tensor:
        return self.feature_extractor.encode_audio(video_path)

    def is_predictable(
        self,
        ) -> bool:
        if self.video_feats is None or self.video_mask is None or self.video_path is None:
            return False
        if self.feature_name == 'clip_slowfast_pann' and self.audio_feats is None:
            return False
        return True

    def prepare_batch(
        self,
        query_feats: torch.Tensor,
        query_mask: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        
        if self.model_name == 'cg_detr':
            model_inputs = dict(
                src_vid=self.video_feats,
                src_vid_mask=self.video_mask,
                src_txt=query_feats,
                src_txt_mask=query_mask,
                src_aud=self.audio_feats,
                vid=None, qid=None
            )
        else:
            model_inputs = dict(
                src_vid=self.video_feats,
                src_vid_mask=self.video_mask,
                src_txt=query_feats,
                src_txt_mask=query_mask,
                src_aud=self.audio_feats
            )
        
        if self.model_name == 'taskweave':
            model_inputs["epoch_i"] = None

        return model_inputs

    def post_processing(
        self,
        inputs: Dict[str, Optional[torch.Tensor]],
        outputs: Dict[str, torch.Tensor],
    ) -> Tuple[List[float], List[float]]:
        prob = F.softmax(outputs["pred_logits"], -1).squeeze(0).cpu()
        scores = prob[:,0]
        pred_spans = outputs["pred_spans"].squeeze(0).cpu()
        
        if self.video_feats is not None:
            video_duration = self.video_feats.shape[1] * self.clip_len
            pred_spans = torch.clamp(span_cxw_to_xx(pred_spans) * video_duration, min=0, max=video_duration)
            cur_ranked_preds = torch.cat([pred_spans, scores[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            saliency_scores = outputs["saliency_scores"][inputs["src_vid_mask"] == 1].cpu().tolist()
        else:
            cur_ranked_preds = []
            saliency_scores = []

        return cur_ranked_preds, saliency_scores

    @torch.no_grad()
    def predict(
        self,
        query: str) -> Optional[Dict[str, List[float]]]:
        is_predictable = self.is_predictable()
        if not is_predictable:
            return None

        # extract text feature
        query_feats: torch.Tensor
        query_mask: torch.Tensor
        query_feats, query_mask = self.feature_extractor.encode_text(query)            
        if self.feature_name != "resnet_glove":
            query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)

        # forward inputs to the model - taskweave returns a tuple of prediction and losses
        inputs = self.prepare_batch(query_feats, query_mask)
        if self.model_name == 'taskweave':
            outputs, _ = self.model(**inputs)
        else:
            outputs = self.model(**inputs)

        ranked_moments, saliency_scores = self.post_processing(inputs, outputs)

        if len(ranked_moments) == 0 and len(ranked_moments) == 0:
            return None
        
        prediction = {
            "pred_relevant_windows": ranked_moments,
            "pred_saliency_scores": saliency_scores,
        }
        return prediction


class MomentDETRPredictor(BasePredictor):
    def __init__(
        self, 
        ckpt_path: str,
        device: str,
        feature_name: str,
        slowfast_path: Optional[str] = None,
        pann_path: Optional[str] = None
        ) -> None:
        super().__init__('moment_detr', ckpt_path, device,
                        feature_name, slowfast_path, pann_path)


class QDDETRPredictor(BasePredictor):
    def __init__(
        self, 
        ckpt_path: str,
        device: str,
        feature_name: str,
        slowfast_path: Optional[str] = None,
        pann_path: Optional[str] = None
        ) -> None:
        super().__init__('qd_detr', ckpt_path, device,
                        feature_name, slowfast_path, pann_path)


class EaTRPredictor(BasePredictor):
    def __init__(
        self, 
        ckpt_path: str,
        device: str,
        feature_name: str,
        slowfast_path: Optional[str] = None,
        pann_path: Optional[str] = None
        ) -> None:
        super().__init__('eatr', ckpt_path, device,
                         feature_name, slowfast_path, pann_path)


class CGDETRPredictor(BasePredictor):
    def __init__(
        self, 
        ckpt_path: str,
        device: str,
        feature_name: str,
        slowfast_path: Optional[str] = None,
        pann_path: Optional[str] = None
        ) -> None:
        super().__init__('cg_detr', ckpt_path, device, 
                         feature_name, slowfast_path, pann_path)


class TRDETRPredictor(BasePredictor):
    def __init__(
        self, 
        ckpt_path: str,
        device: str,
        feature_name: str,
        slowfast_path: Optional[str] = None,
        pann_path: Optional[str] = None
        ) -> None:
        super().__init__('tr_detr', ckpt_path, device,
                         feature_name, slowfast_path, pann_path)


class UVCOMPredictor(BasePredictor):
    def __init__(
        self, 
        ckpt_path: str,
        device: str,
        feature_name: str,
        slowfast_path: Optional[str] = None,
        pann_path: Optional[str] = None
        ) -> None:
        super().__init__('uvcom', ckpt_path, device,
                         feature_name, slowfast_path, pann_path)


class TaskWeavePredictor(BasePredictor):
    def __init__(
        self, 
        ckpt_path: str,
        device: str,
        feature_name: str,
        slowfast_path: Optional[str] = None,
        pann_path: Optional[str] = None
        ) -> None:
        super().__init__('taskweave', ckpt_path, device,
                         feature_name, slowfast_path, pann_path)