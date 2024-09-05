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

import os
import time
import torch
import argparse
import shutil
import yaml

from lighthouse.common.utils.basic_utils import mkdirp, load_json, save_json, make_zipfile, dict_to_markdown
from easydict import EasyDict

class BaseOptions(object):
    def __init__(self, model, dataset, feature):
        self.model = model
        self.dataset = dataset
        self.feature = feature
        self.opt = {}

    @property
    def option(self):
        if len(self.opt) == 0:
            raise RuntimeError('option is empty. Did you run parse()?')
        return self.opt

    def update(self, yaml_file):
        with open(yaml_file, 'r') as f:
            yml = yaml.load(f, Loader=yaml.FullLoader)
            self.opt.update(yml)

    def parse(self):
        base_cfg = 'configs/base.yml'
        feature_cfg = f'configs/feature/{self.feature}.yml'
        model_cfg = f'configs/model/{self.model}.yml'
        dataset_cfg = f'configs/dataset/{self.dataset}.yml'
        cfgs = [base_cfg, feature_cfg, model_cfg, dataset_cfg]
        for cfg in cfgs:
            self.update(cfg)

        self.opt = EasyDict(self.opt)

        # result directory
        self.opt.results_dir = os.path.join(self.opt.results_dir, self.model, self.dataset, self.feature)
        self.opt.ckpt_filepath = os.path.join(self.opt.results_dir, self.opt.ckpt_filename)
        self.opt.train_log_filepath = os.path.join(self.opt.results_dir, self.opt.train_log_filename)
        self.opt.eval_log_filepath = os.path.join(self.opt.results_dir, self.opt.eval_log_filename)

        # feature directory
        v_feat_dirs = None
        t_feat_dir = None
        a_feat_dirs = None
        a_feat_types = None

        if self.feature == 'clip_slowfast_pann':
            v_feat_dirs = [f'features/{self.dataset}/clip', f'features/{self.dataset}/slowfast']
            t_feat_dir = f'features/{self.dataset}/clip_text'
            a_feat_dirs = [f'features/{self.dataset}/pann']
            a_feat_types = self.opt.a_feat_types
            
        elif self.feature == 'clip_slowfast':
            v_feat_dirs = [f'features/{self.dataset}/clip', f'features/{self.dataset}/slowfast']
            t_feat_dir = f'features/{self.dataset}/clip_text'

        elif self.feature == 'clip':
            v_feat_dirs = [f'features/{self.dataset}/clip']
            t_feat_dir = f'features/{self.dataset}/clip_text'

        elif self.feature == 'resnet_glove':
            v_feat_dirs = [f'features/{self.dataset}/resnet']
            t_feat_dir = f'features/{self.dataset}/glove'

        elif self.feature == 'i3d_clip':
            v_feat_dirs = [f'features/{self.dataset}/i3d']
            t_feat_dir = f'features/{self.dataset}/clip_text'

        self.opt.v_feat_dirs = v_feat_dirs
        self.opt.t_feat_dir = t_feat_dir
        self.opt.a_feat_dirs = a_feat_dirs
        self.opt.a_feat_types = a_feat_types
    
    def makedirs(self):
        if 'results_dir' not in self.opt:
            raise RuntimeError('results_dir is not set in self.opt. Did you run parse()?')
        os.makedirs(self.opt.results_dir, exist_ok=True)
        if 'domains' in self.opt:
            for domain in self.domains:
                os.makedirs(os.path.join(self.opt.results_dir, domain), exist_ok=True)