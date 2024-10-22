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

from lighthouse.models import QDDETRPredictor
from typing import Dict, List, Optional

def load_weights(weight_dir: str) -> None:
    if not os.path.exists(os.path.join(weight_dir, 'clap_qd_detr_clotho-moment.ckpt')):  
        command = 'wget -P gradio_demo/weights/ https://zenodo.org/records/13961029/files/clap_qd_detr_clotho-moment.ckpt'
        subprocess.run(command, shell=True)

# use GPU if available
device: str = 'cpu'
weight_dir: str = 'gradio_demo/weights'
weight_path: str = os.path.join(weight_dir, 'clap_qd_detr_clotho-moment.ckpt')
load_weights(weight_dir)
model: QDDETRPredictor = QDDETRPredictor(weight_path, device=device, feature_name='clap')

# encode audio features
model.encode_audio('api_example/1a-ODBWMUAE.wav')

# moment retrieval
query: str = 'Water cascades down from a waterfall.'
prediction: Optional[Dict[str, List[float]]] = model.predict(query)
print(prediction)
