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

from lighthouse.common.utils.basic_utils import mkdirp, load_json, save_json, make_zipfile, dict_to_markdown
import shutil
import yaml

from easydict import EasyDict

class BaseOptions(object):
    def __init__(self):
        pass

    def parse(self, yaml_path):
        opt = {}

        # base yaml
        with open('configs/base.yml', 'r') as f:
            yml = yaml.load(f, Loader=yaml.FullLoader)
            opt.update(yml)
        
        with open('{}'.format(yaml_path), 'r') as f:
            yml = yaml.load(f, Loader=yaml.FullLoader)
            opt.update(yml)

        opt = EasyDict(opt)
        opt.ckpt_filepath = os.path.join(opt.results_dir, opt.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, opt.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, opt.eval_log_filename)
        os.makedirs(opt.results_dir, exist_ok=True)        
        return opt