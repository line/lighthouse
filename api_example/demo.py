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
import subprocess
import pprint
import torch
from lighthouse.models import CGDETRPredictor

# use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

weight_dir = 'gradio_demo/weights'
if not os.path.exists(os.path.join(weight_dir, 'clip_cg_detr_qvhighlight.ckpt')):  
    command = 'wget -P gradio_demo/weights/ https://zenodo.org/records/13363606/files/clip_cg_detr_qvhighlight.ckpt'
    subprocess.run(command, shell=True)

if not os.path.exists('SLOWFAST_8x8_R50.pkl'):
    subprocess.run('wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl')

# slowfast_path is necesary if you use clip_slowfast features
weight_path = os.path.join(weight_dir, 'clip_cg_detr_qvhighlight.ckpt')
query = 'A woman wearing a glass is speaking in front of the camera'
model = CGDETRPredictor(weight_path, device=device, feature_name='clip', slowfast_path='SLOWFAST_8x8_R50.pkl')

# encode video features
model.encode_video('api_example/RoripwjYFp8_60.0_210.0.mp4')

# moment retrieval & highlight detection
prediction = model.predict(query)
pprint.pp(prediction)