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

from lighthouse.models import MarengoPredictor
from typing import Dict, List, Optional

# Zero-shot moment retrieval with TwelveLabs Marengo embeddings.
# No local checkpoint or feature files are needed; the video is embedded
# server-side. Set TWELVELABS_API_KEY first (free key: https://twelvelabs.io).
api_key: Optional[str] = os.environ.get('TWELVELABS_API_KEY')
model: MarengoPredictor = MarengoPredictor(api_key=api_key, clip_length=2.0)

# encode video clips (local file or a public URL)
video: Dict[str, List[List[float]]] = model.encode_video('api_example/RoripwjYFp8_60.0_210.0.mp4')

# moment retrieval
query: str = 'A woman wearing a glass is speaking in front of the camera'
prediction: Optional[Dict[str, List[List[float]]]] = model.predict(query, video)
print(prediction)
