# Lighthouse

![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/awkrail/lighthouse_demo)

Lighthouse is a user-friendly library for reproducible video moment retrieval and highlight detection (MR-HD).
It supports seven models, four features (video and audio features), and six datasets for reproducible MR-HD, MR, and HD. In addition, we prepare an inference API and Gradio demo for developers to use state-of-the-art MR-HD approaches easily.

**News**: Our demo paper is available on arXiv. Any comments are welcome: [Lighthouse: A User-Friendly Library for Reproducible Video Moment Retrieval and Highlight Detection](https://www.arxiv.org/abs/2408.02901)

## Installation
Install ffmpeg first. If you are an Ubuntu user, run:
```
apt install ffmpeg
```
Then, install pytorch, torchvision, torchaudio, and torchtext based on your GPU environments.
Note that the inference API is available for CPU environments. We tested the codes on Python 3.9 and CUDA 11.7:
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 torchtext==0.15.1
```
Finally, run to install dependency libraries:
```
pip install 'git+https://github.com/line/lighthouse.git'
```

## Inference API (Available for both CPU/GPU mode)
Lighthouse supports the following inference API:
```python
import torch
from lighthouse.models import CGDETRPredictor

# use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# slowfast_path is necesary if you use clip_slowfast features
query = 'A man is speaking in front of the camera'
model = CGDETRPredictor('results/clip_slowfast_cg_detr/qvhighlight/best.ckpt', device=device,
                        feature_name='clip_slowfast', slowfast_path='SLOWFAST_8x8_R50.pkl')

# encode video features
model.encode_video('api_example/RoripwjYFp8_60.0_210.0.mp4')

# moment retrieval & highlight detection
prediction = model.predict(query)
print(prediction)
"""
pred_relevant_windows: [[start, end, score], ...,]
pred_saliency_scores: [score, ...]

{'query': 'A man is speaking in front of the camera',
 'pred_relevant_windows': [[117.1296, 149.4698, 0.9993],
                           [-0.1683, 5.4323, 0.9631],
                           [13.3151, 23.42, 0.8129],
                           ...],
 'pred_saliency_scores': [-10.868017196655273,
                          -12.097496032714844,
                          -12.483806610107422,
                          ...]}
"""
```
Run `python api_example/demo.py` to reproduce the results. It automatically downloads pre-trained weights for CG-DETR (CLIP backbone). If you want to use other models, download [pre-trained weights](https://drive.google.com/file/d/1ebQbhH1tjgTmRBmyOoW8J9DH7s80fqR9/view?usp=drive_link) 
In addition, to use `clip_slowfast` features, it is necessary to download slowfast pre-trained weights ([SLOWFAST_8x8_R50](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl)).

**Limitation**: The maximum video duration is **150s** due to the current benchmark datasets. Using CLIP+Slowfast feature (`feature_name='clip_slowfast'`) is possible for CPU users, but very slow. Use `feature_name='clip'`.

## Gradio demo
Run `python gradio_demo/demo.py`. Upload the video and input text query, and click the blue button.

![Gradio demo image](images/demo_improved.png)

## Supported models, datasets, and features
### Models
Moment retrieval & highlight detection
- [x] : [Moment-DETR (Lei et al. NeurIPS21)](https://arxiv.org/abs/2107.09609)
- [x] : [QD-DETR (Moon et al. CVPR23)](https://arxiv.org/abs/2303.13874)
- [x] : [EaTR (Jang et al. ICCV23)](https://arxiv.org/abs/2308.06947)
- [x] : [CG-DETR (Moon et al. arXiv24)](https://arxiv.org/abs/2311.08835)
- [x] : [UVCOM (Xiao et al. CVPR24)](https://arxiv.org/abs/2311.16464)
- [x] : [TR-DETR (Sun et al. AAAI24)](https://arxiv.org/abs/2401.02309)
- [x] : [TaskWeave (Jin et al. CVPR24)](https://arxiv.org/abs/2404.09263)

### Datasets
Moment retrieval & highlight detection
- [x] : [QVHighlights (Lei et al. NeurIPS21)](https://arxiv.org/abs/2107.09609)
- [x] : [QVHighlights w/ Audio Features (Lei et al. NeurIPS21)](https://arxiv.org/abs/2107.09609)
- [ ] : [QVHighlights ASR Pretraining (Lei et al. NeurIPS21)](https://arxiv.org/abs/2107.09609)

Moment retrieval
- [x] : [ActivityNet Captions (Krishna et al. ICCV17)](https://arxiv.org/abs/1705.00754)
- [x] : [Charades-STA (Gao et al. ICCV17)](https://arxiv.org/abs/1705.02101)
- [x] : [TaCoS (Regneri et al. TACL13)](https://aclanthology.org/Q13-1003/)
- [ ] : [DiDeMo (Hendricks et al. EMNLP18)](https://arxiv.org/abs/1809.01337)

Highlight detection
- [x] : [TVSum (Song et al. CVPR15)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Song_TVSum_Summarizing_Web_2015_CVPR_paper.pdf)
- [x] : [YouTube Highlights (Sun et al. ECCV14)](https://grail.cs.washington.edu/wp-content/uploads/2015/08/sun2014rdh.pdf)
- [ ] : [TVSum Audio Training (Song et al. CVPR15)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Song_TVSum_Summarizing_Web_2015_CVPR_paper.pdf)

### Features
- [x] : ResNet+GloVe
- [x] : CLIP
- [x] : CLIP+Slowfast
- [ ] : CLIP+Slowfast+PANNs (Audio) for QVHighlights
- [x] : I3D+CLIP (Text) for TVSum

## Reproduce the experiments

### Pre-trained weights
Pre-trained weights can be downloaded from [here](https://drive.google.com/file/d/1ebQbhH1tjgTmRBmyOoW8J9DH7s80fqR9/view?usp=drive_link).
Download and unzip on the home directory. If you want individual weights, download from [reproduced results tables](#reproduced-results).

### Datasets
Due to the copyright issue, we here distribute only feature files.
Download and place them under `./features` directory.
To extract features from videos, we use [HERO_Video_Feature_Extractor](https://github.com/linjieli222/HERO_Video_Feature_Extractor).

- [QVHighlights](https://drive.google.com/file/d/1-ALnsXkA4csKh71sRndMwybxEDqa-dM4/view?usp=sharing)
- [Charades-STA](https://drive.google.com/file/d/1EOeP2A4IMYdotbTlTqDbv5VdvEAgQJl8/view?usp=sharing)
- [ActivityNet Captions](https://drive.google.com/file/d/1P2xS998XfbN5nSDeJLBF1m9AaVhipBva/view?usp=sharing)
- [TACoS](https://drive.google.com/file/d/1rYzme9JNAk3niH1K81wgT13pOMn005jb/view?usp=sharing)
- [TVSum](https://drive.google.com/file/d/1gSex1hpXLxHQu6zHyyQISKZjP7Ndt6U9/view?usp=sharing)
- [YouTube Highlight](https://drive.google.com/file/d/12swoymGwuN5TlDlWBTo6UUWVm2DqVBpn/view?usp=sharing)

The whole directory should be look like this:
```
lighthouse/
├── api_example
├── configs
├── data
├── features # Download the features and place them here
│   ├── ActivityNet
│   │   ├── clip
│   │   ├── clip_text
│   │   ├── resnet
│   │   └── slowfast
│   ├── Charades
│   │   ├── clip
│   │   ├── clip_text
│   │   ├── resnet
│   │   ├── slowfast
│   │   └── vgg
│   ├── QVHighlight
│   │   ├── clip
│   │   ├── clip_text
│   │   ├── pann
│   │   ├── resnet
│   │   └── slowfast
│   ├── tacos
│   │   ├── clip
│   │   ├── clip_text
│   │   ├── meta
│   │   ├── resnet
│   │   └── slowfast
│   ├── tvsum
│   │   ├── audio
│   │   ├── clip
│   │   ├── clip_text
│   │   ├── i3d
│   │   ├── resnet
│   │   ├── slowfast
│   │   └── tvsum_anno.json
│   └── youtube_highlight
│       ├── clip
│       ├── clip_text
│       └── slowfast
├── gradio_demo
├── images
├── lighthouse
├── results # The pre-trained weights are saved in this directory
└── training
```

### Training and evaluation

#### Training
The general training command is:
```
PYTHONPATH=. python training/train.py --config configs/DATASET/FEATURE_MODEL_DATASET.yml
```
|         | Options                                                            |
|---------|--------------------------------------------------------------------|
| Model   | moment_detr, qd_detr, eatr, cg_detr, uvcom, tr_detr, taskweave     |
| Feature | resnet_glove, clip, clip_slowfast, clip_slowfast_pann              |
| Dataset | qvhighlight, activitynet, charades, tacos, tvsum, youtube_highlight|

For example, to train moment_detr on QVHighlights with CLIP+Slowfast features, run:
```
PYTHONPATH=. python training/train.py --config configs/qvhighlight/clip_slowfast_moment_detr_qvhighlight.yml
```
To train the models on HD datasets (i.e., TVSum and YouTube Highlight), you need to specify the domain.<br>
For example, to train moment_detr in BK domain on TVSum, run:
```
PYTHONPATH=. python training/train.py --config configs/tvsum/clip_slowfast_moment_detr_tvsum.yml --domain BK
```

#### Evaluation
The evaluation command is (in this example, we evaluate QD-DETR/CLIP+Slowfast on the QVHighlight val set):
```
PYTHONPATH=. python training/evaluate.py --config configs/qvhighlight/clip_slowfast_qd_detr_qvhighlight.yml \ 
                                         --model_path results/clip_slowfast_qd_detr/qvhighlight/best.ckpt \
                                         --eval_split_name val \
                                         --eval_path data/qvhighlight/highlight_val_release.jsonl
```
To generate submission files for QVHighlight test sets, run (**QVHighlights only**):
```
PYTHONPATH=. python training/evaluate.py --config configs/qvhighlight/clip_slowfast_qd_detr_qvhighlight.yml \ 
                                         --model_path results/clip_slowfast_qd_detr/qvhighlight/best.ckpt \
                                         --eval_split_name test \
                                         --eval_path data/qvhighlight/highlight_test_release.jsonl
```
Then zip `hl_val_submission.jsonl` and `hl_test_submission.jsonl`, and submit it to the [Codalab](https://codalab.lisn.upsaclay.fr/competitions/6937) (**QVHighlights only**):
```
zip -r submission.zip val_submission.jsonl test_submission.jsonl
```

## Reproduced results
See [here](markdown/reproduced_results.md). You can download individual checkpoints.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Todo
- [ ] : Support Wandb
- [ ] : Support DeDiMo
- [ ] : Support ASR-based pretraining on QVHighlights

## LICENSE
Apache License 2.0

## Contact
Taichi Nishimura ([taichitary@gmail.com](taichitary@gmail.com))
