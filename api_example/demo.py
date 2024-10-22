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
import torch

from lighthouse.models import CGDETRPredictor
from typing import Dict, List, Optional

def load_weights(weight_dir: str) -> None:
    if not os.path.exists(os.path.join(weight_dir, 'clip_slowfast_pann_cg_detr_qvhighlight.ckpt')):  
        command = 'wget -P gradio_demo/weights/ https://zenodo.org/records/13960580/files/clip_slowfast_pann_cg_detr_qvhighlight.ckpt'
        subprocess.run(command, shell=True)

    if not os.path.exists('SLOWFAST_8x8_R50.pkl'):
        subprocess.run('wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl', shell=True)

    if not os.path.exists('Cnn14_mAP=0.431.pth'):
        subprocess.run('wget https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth', shell=True)

# use GPU if available
device: str = 'cpu'
weight_dir: str = 'gradio_demo/weights'
weight_path: str = os.path.join(weight_dir, 'clip_slowfast_cg_detr_qvhighlight.ckpt')
load_weights(weight_dir)
model: CGDETRPredictor = CGDETRPredictor(weight_path, device=device, feature_name='clip_slowfast', 
                                         slowfast_path='SLOWFAST_8x8_R50.pkl', pann_path=None)

# encode video features
model.encode_video('api_example/RoripwjYFp8_60.0_210.0.mp4')

# moment retrieval & highlight detection
query: str = 'A woman wearing a glass is speaking in front of the camera'
prediction: Optional[Dict[str, List[float]]] = model.predict(query)
print(prediction)
