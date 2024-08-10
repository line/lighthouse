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
import math
import clip
import torch
import torchvision

from torchtext import vocab

from lighthouse.video_loader import VideoLoader, SlowfastVideoReader, clip_iterator, pack_pathway_output
from lighthouse.slowfast.model import slowfast_model_loader


class Preprocessing(object):

    def __init__(self, feature_name):
        if feature_name == 'resnet_glove':
            self.norm = Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        else:
            self.norm = Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])

    def __call__(self, tensor):
        tensor = tensor / 255.0
        tensor = self.norm(tensor)
        return tensor


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = torch.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


class SlowFastNormalize(object):

    def __init__(self, mean, std, device):
        self.mean = torch.FloatTensor(mean).view(1, 3, 1, 1, 1).float().to(device)
        self.std = torch.FloatTensor(std).view(1, 3, 1, 1, 1).float().to(device)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


class GlobalAvgPool(torch.nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=[-2, -1])


class VideoFeatureExtractor:
    def __init__(self, framerate, size, centercrop, feature_name, device, slowfast_path):
        if feature_name == 'clip_slowfast':
            self.slowfast_feature_dim = 2304
            self.slowfast_norm = SlowFastNormalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], device=device)
            self.sf_loader = SlowfastVideoReader(framerate=30, size=size, clip_len=1/framerate, centercrop=centercrop)
            self.video_loader = VideoLoader(framerate=framerate, size=size, centercrop=centercrop)
            self.clip_extractor, _ = clip.load('ViT-B/32', device=device, jit=False)
            self.slowfast_extractor = slowfast_model_loader(slowfast_path, device=device).eval()
        
        elif feature_name == 'clip':
            self.video_loader = VideoLoader(framerate=framerate, size=size, centercrop=centercrop)
            self.clip_extractor, _ = clip.load('ViT-B/32', device=device, jit=False)
        
        elif feature_name == 'resnet_glove':
            resnet_model = torchvision.models.resnet152(pretrained=True)
            self.video_loader = VideoLoader(framerate=framerate, size=size, centercrop=centercrop)
            self.resnet_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-2], GlobalAvgPool()).eval()
            self.resnet_extractor.to(device)

            # load GloVe
            self.vocab = vocab.pretrained_aliases['glove.6B.300d']()
            self.vocab.itos.extend(['<unk>'])
            self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
            self.vocab.vectors = torch.cat(
                (self.vocab.vectors, torch.zeros(1, self.vocab.dim)), dim=0)
            self.embedding = torch.nn.Embedding.from_pretrained(self.vocab.vectors)
        
        else:
            raise NotImplementedError()

        self.feature_name = feature_name
        self.tokenizer = clip.tokenize
        self.video_preprocessor = Preprocessing(feature_name)
        self.device = device
    

    @torch.no_grad()
    def extract_clip_feature(self, video_frames, bsz=60):
        # CLIP
        video_frames = self.video_preprocessor(video_frames)
        n_frames = len(video_frames)
        n_batch = int(math.ceil(n_frames / bsz))
        video_features = []
        for i in range(n_batch):
            st_idx = i * bsz
            ed_idx = (i+1) * bsz
            _video_frames = video_frames[st_idx:ed_idx].to(self.device)
            _video_features = self.clip_extractor.encode_image(_video_frames)
            video_features.append(_video_features)
        video_features = torch.cat(video_features, dim=0)
        return video_features


    @torch.no_grad()
    def extract_resnet_feature(self, video_frames, bsz=60):
        # ResNet
        # TODO: implement resnet + glove features
        video_frames = self.video_preprocessor(video_frames)
        n_frames = len(video_frames)
        n_batch = int(math.ceil(n_frames / bsz))
        video_features = []
        for i in range(n_batch):
            st_idx = i * bsz
            ed_idx = (i+1) * bsz
            _video_frames = video_frames[st_idx:ed_idx].to(self.device)
            _video_features = self.resnet_extractor(_video_frames)
            _video_features = torch.nn.functional.normalize(_video_features, dim=-1)
            video_features.append(_video_features)
        video_features = torch.cat(video_features, dim=0)
        return video_features


    @torch.no_grad()
    def extract_slowfast_feature(self, slowfast_frames, bsz=45):
        n_chunk = len(slowfast_frames)
        features = torch.HalfTensor(n_chunk, self.slowfast_feature_dim, device=slowfast_frames.device).fill_(0)
        n_batch = int(math.ceil(n_chunk / bsz))
        for i in range(n_batch):
            st_idx = i * bsz
            ed_idx = (i+1) * bsz
            fast_clip = slowfast_frames[st_idx:ed_idx].float().to(self.device)
            fast_clip = fast_clip.permute(0, 4, 1, 2, 3)
            fast_clip = fast_clip/255.
            fast_clip = self.slowfast_norm(fast_clip)
            inputs = pack_pathway_output(fast_clip, self.device)
            batch_features = self.slowfast_extractor(inputs)
            features[st_idx:ed_idx] = batch_features.half()
        slowfast_features = features.cpu()
        return slowfast_features


    @torch.no_grad()
    def encode_video(self, video_path, bsz=60):
        if self.feature_name == 'clip_slowfast':
            video_frames = self.video_loader.read_video_from_file(video_path)  # (T, H, W, 3)
            slowfast_frames = self.sf_loader.read_video_from_file(video_path)
            
            clip_video_features = self.extract_clip_feature(video_frames)
            slowfast_features = self.extract_slowfast_feature(slowfast_frames)

            clip_video_features = clip_video_features.to(self.device)
            slowfast_features = slowfast_features.to(self.device)   

            # trim the number of frames for smaller frames
            smaller_frame_len = min(clip_video_features.shape[0], slowfast_features.shape[0])
            clip_video_features = clip_video_features[:smaller_frame_len]
            slowfast_features = slowfast_features[:smaller_frame_len]

            clip_slowfast_features = torch.cat([clip_video_features, slowfast_features], dim=-1)
            return clip_slowfast_features

        elif self.feature_name == 'clip':
            video_frames = self.video_loader.read_video_from_file(video_path)  # (T, H, W, 3)
            video_features = self.extract_clip_feature(video_frames)
            video_features = video_features.to(self.device)
            return video_features  # (T=#frames, d) torch tensor

        elif self.feature_name == 'resnet_glove':
            video_frames = self.video_loader.read_video_from_file(video_path)  # (T, H, W, 3)
            video_features = self.extract_resnet_feature(video_frames)
            video_features = video_features.to(self.device)
            return video_features
        
        else:
            raise NotImplementedError()

    @property
    def dtype(self):
        return self.clip_extractor.visual.conv1.weight.dtype

    @torch.no_grad()
    def encode_text(self, query, bsz=60):
        if self.feature_name == 'resnet_glove':
            word_inds = torch.LongTensor(
                [self.vocab.stoi.get(w.lower(), 400000) for w in query.split()])
            mask = torch.ones((1, word_inds.shape[0])).to(self.device)
            return self.embedding(word_inds).unsqueeze(0).to(self.device), mask
        else:
            text = self.tokenizer(query).to(self.device)
            x = self.clip_extractor.token_embedding(text).type(self.dtype)
            x = x + self.clip_extractor.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_extractor.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip_extractor.ln_final(x).type(torch.float32)
            mask = (text != 0).type(torch.float32).to(self.device)
            return x, mask