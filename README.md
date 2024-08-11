# lighthouse

![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

Lighthouse is a user-friendly library for reproducible video moment retrieval and highlight detection (MR-HD).
It supports six models, three features, and five datasets for reproducible MR-HD, MR, and HD. In addition, we prepare an inference-only API and Gradio demo for developers to use state-of-the-art MR-HD approaches easily.

**News**: Our demo paper is available on arXiv. Any comments are welcome: [Lighthouse: A User-Friendly Library for Reproducible Video Moment Retrieval and Highlight Detection](https://www.arxiv.org/abs/2408.02901)

## Installation
Install pytorch, torchvision, torchaudio, and torchtext based on your GPU environments.
Note that the inference API is available for CPU environments. We tested the codes on Python 3.9 and CUDA 11.7:
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 torchtext==0.15.1
```
Then run:
```
pip install .
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
model = CGDETRPredictor('results/clip_slowfast_cg_detr/activitynet/best.ckpt', device=device, 
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
Download [pre-trained weights](https://drive.google.com/file/d/1ebQbhH1tjgTmRBmyOoW8J9DH7s80fqR9/view?usp=drive_link) and Run `python api_example/demo.py` to reproduce the results. In addition, to use `clip_slowfast` features, it is necessary to download slowfast pre-trained weights ([SLOWFAST_8x8_R50](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl)).

**Limitation**: The maximum video duration is **150s** due to the current benchmark datasets. Running the code on CPU is possible, but very slow. Use `feature_name='clip'` for CPU users.

## Gradio Demo
Run `python gradio_demo/demo.py`. Upload the video and input text query, and click the blue button.

![Gradio demo image](images/demo_improved.png)

## Supported models and datasets
### Models
- [x] : [Moment-DETR (Lei et al. NeurIPS21)](https://arxiv.org/abs/2107.09609)
- [x] : [QD-DETR (Moon et al. CVPR23)](https://arxiv.org/abs/2303.13874)
- [x] : [EaTR (Jang et al. ICCV23)](https://arxiv.org/abs/2308.06947)
- [x] : [CG-DETR (Moon et al. arXiv24)](https://arxiv.org/abs/2311.08835)
- [x] : [UVCOM (Xiao et al. CVPR24)](https://arxiv.org/abs/2311.16464)
- [x] : [TR-DETR (Sun et al. AAAI24)](https://arxiv.org/abs/2401.02309)
- [ ] : [TaskWeave (Jin et al. CVPR24)](https://arxiv.org/abs/2404.09263)

### Datasets
Moment retrieval & highlight detection
- [x] : [QVHighlights (Lei et al. NeurIPS21)](https://arxiv.org/abs/2107.09609)
- [ ] : [QVHighlights Audio Pretraining (Lei et al. NeurIPS21)](https://arxiv.org/abs/2107.09609)

Moment retrieval
- [x] : [ActivityNet Captions (Krishna et al. ICCV17)](https://arxiv.org/abs/1705.00754)
- [x] : [Charades-STA (Gao et al. ICCV17)](https://arxiv.org/abs/1705.02101)
- [x] : [TaCoS (Regneri et al. TACL13)](https://aclanthology.org/Q13-1003/)
- [ ] : [DiDeMo (Hendricks et al. EMNLP18)](https://arxiv.org/abs/1809.01337)

Highlight detection
- [x] : [TVSum (Song et al. CVPR15)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Song_TVSum_Summarizing_Web_2015_CVPR_paper.pdf)
- [ ] : [TVSum Audio Training (Song et al. CVPR15)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Song_TVSum_Summarizing_Web_2015_CVPR_paper.pdf)
- [ ] : [YouTube Highlights (Xu et al. ICCV21)](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Cross-Category_Video_Highlight_Detection_via_Set-Based_Learning_ICCV_2021_paper.pdf)

### Features
- [x] : ResNet+GloVe
- [x] : CLIP
- [x] : CLIP+Slowfast
- [x] : I3D+CLIP(Text) for TVSum

## Reproduce the Experiments

### Pre-trained weights
Pre-trained weights can be downloaded from [here](https://drive.google.com/file/d/1ebQbhH1tjgTmRBmyOoW8J9DH7s80fqR9/view?usp=drive_link).
Download and unzip on the home directory.

### Datasets
Due to the copyright issue, we here distribute only feature files.
Download and place them under `./features` directory.
To extract features from videos, we use [HERO_Video_Feature_Extractor](https://github.com/linjieli222/HERO_Video_Feature_Extractor).

- [QVHighlights](https://drive.google.com/file/d/1-ALnsXkA4csKh71sRndMwybxEDqa-dM4/view?usp=sharing)
- [Charades-STA](https://drive.google.com/file/d/1EOeP2A4IMYdotbTlTqDbv5VdvEAgQJl8/view?usp=sharing)
- [ActivityNet Captions](https://drive.google.com/file/d/1P2xS998XfbN5nSDeJLBF1m9AaVhipBva/view?usp=sharing)
- [TACoS](https://drive.google.com/file/d/1rYzme9JNAk3niH1K81wgT13pOMn005jb/view?usp=sharing)
- [TVSum](https://drive.google.com/file/d/1gSex1hpXLxHQu6zHyyQISKZjP7Ndt6U9/view?usp=sharing)

The whole directory should be look like this:
```
lighthouse/
└── features
    ├── ActivityNet
    │   ├── clip
    │   ├── clip_text
    │   ├── resnet
    │   └── slowfast
    ├── Charades
    │   ├── clip
    │   ├── clip_text
    │   ├── resnet
    │   ├── slowfast
    │   └── vgg
    ├── QVHighlight
    │   ├── clip
    │   ├── clip_text
    │   ├── pann
    │   ├── resnet
    │   └── slowfast
    ├── tacos
    │   ├── clip
    │   ├── clip_text
    │   ├── meta
    │   ├── resnet
    │   └── slowfast
    └── tvsum
        ├── audio
        ├── clip
        ├── clip_text
        ├── i3d
        ├── resnet
        ├── slowfast
        └── tvsum_anno.json
```

### Training and Evaluation
The general training command is:
```
PYTHONPATH="." python training/train.py --config configs/DATASET/FEATURE_MODEL_DATASET.yml
```

|         | Options                                                   |
|---------|-----------------------------------------------------------|
| Model   | moment_detr, qd_detr, eatr, cg_detr, uvcom, tr_detr       |
| Feature | resnet_glove, clip, clip_slowfast                         |
| Dataset | qvhighlight, activitynet, charades, tacos, tvsum          |

To train moment_detr on QVHighlights with CLIP+Slowfast features, run:
```
PYTHONPATH="." python training/train.py --config configs/qvhighlight/clip_slowfast_moment_detr_qvhighlight.yml
```

The evaluation command is like (In this example, we evaluate QD-DETR on the Charades-STA dataset):
```
PYTHONPATH="." python training/evaluate.py --config configs/charades/clip_slowfast_qd_detr_charades.yml \
                                           --model_path results/clip_slowfast_qd_detr/charades/best.ckpt \
                                           --eval_split_name val \
                                           --eval_path data/charades/charades_test_release.jsonl
```
To generate submission files for QVHighlight test sets, run (**QVHighlights only**):
```
PYTHONPATH="." python training/evaluate.py --config configs/qvhighlight/clip_slowfast_qd_detr_qvhighlight.yml \ 
                                           --model_path results/clip_slowfast_qd_detr/qvhighlight/best.ckpt \
                                           --eval_split_name test \
                                           --eval_path data/qvhighlight/highlight_test_release.jsonl
```
Then zip `hl_val_submission.jsonl` and `hl_test_submission.jsonl`, and submit it to the [Codalab](https://codalab.lisn.upsaclay.fr/competitions/6937) (**QVHighlights only**):
```
zip -r submission.zip val_submission.jsonl test_submission.jsonl
```

## Reproduced Results

### QVHighlights (Moment retrieval & highlight detection)
Test set scores are reported.

#### ResNet152+GloVe
|    Models   |  R1@0.5  |  R1@0.7  |  HD mAP  |   HIT@1  | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   40.0   |   22.0   |   30.0   |   42.9   |            |
|   QD-DETR   |   52.7   |   36.1   |   33.8   |   50.7   |            |
|     EaTR    | **57.2** | **38.9** | **36.3** | **57.4** |            |
|   TR-DETR   |   47.7   |   31.6   |   34.3   |   52.0   |            |
|    UVCOM    |   53.8   |   37.6   |   34.8   |   53.8   |            |
|   CG-DETR   |   53.1   |   38.3   |   34.5   |   52.9   |            |

#### CLIP
|    Models   |  R1@0.5  |  R1@0.7  |  HD mAP  |   HIT@1  | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   55.8   |   33.8   |   35.7   |   55.8   |            |
|   QD-DETR   |   60.8   |   41.8   |   38.2   |   60.7   |            |
|     EaTR    |   54.6   |   34.0   |   34.9   |   54.7   |            |
|   TR-DETR   |   60.2   |   41.4   |   38.6   |   59.3   |            |
|    UVCOM    |   62.7   | **46.9** | **39.8** | **64.5** |            |
|   CG-DETR   | **64.5** |   46.0   |   39.4   |   64.3   |            |

#### CLIP+Slowfast
|    Models   |  R1@0.5  |  R1@0.7  |  HD mAP  |   HIT@1  | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   54.4   |   33.9   |   32.6   |   56.7   |            |
|   QD-DETR   |   62.1   |   44.6   |   38.8   |   61.6   |            |
|     EaTR    |   57.2   |   38.9   |   36.6   |   57.9   |            |
|   TR-DETR   | **65.2** | **48.8** |   39.8   |   62.1   |            |
|    UVCOM    |   62.6   |   47.6   |   39.6   |   62.8   |            |
|   CG-DETR   |   64.9   |   48.1   | **40.7** | **67.0** |            |

### ActivityNet Captions (Moment retrieval)
Val_1 scores are reported.

#### ResNet152+GloVe
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   34.2   |   19.5   |   46.3   |   24.4   |            |
|   QD-DETR   |   35.4   |   20.3   |   47.4   |   24.9   |            |
|     EaTR    |   32.4   |   18.2   |   44.3   |   21.9   |            |
|    UVCOM    |   34.4   |   19.9   |   46.1   |   24.4   |            |
|   CG-DETR   | **37.0** | **21.2** | **48.6** | **26.5** |            |

#### CLIP
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   36.1   |   20.4   |   48.2   |   25.7   |            |
|   QD-DETR   |   36.9   |   21.4   |   48.4   |   26.3   |            |
|     EaTR    |   34.6   |   19.7   |   45.1   |   23.1   |            |
|    UVCOM    |   37.0   |   21.5   |   48.3   |   25.7   |            |
|   CG-DETR   | **38.8** | **22.6** | **50.6** | **27.5** |            |

#### CLIP+Slowfast
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   36.5   |   21.1   |   48.4   |   26.0   |            |
|   QD-DETR   |   37.5   |   22.1   |   48.9   |   26.4   |            |
|     EaTR    |   34.6   |   19.3   |   45.2   |   22.3   |            |
|    UVCOM    |   37.3   |   21.6   |   48.9   |   25.7   |            |
|   CG-DETR   | **40.0** | **23.2** | **51.0** | **27.7** |            |

#### Charades-STA (Moment retrieval)
Test set scores are reported.

#### ResNet152+GloVe
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   38.4   |   22.9   |   52.4   |   22.2   |            |
|   QD-DETR   | **42.1** | **24.0** |   56.7   | **24.5** |            |
|     EaTR    |   37.6   |   20.1   |   53.5   |   23.6   |            |
|    UVCOM    |   38.1   |   18.2   |   54.4   |   21.1   |            |
|   CG-DETR   |   39.7   |   19.4   | **56.9** |   23.2   |            |

#### CLIP
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   47.9   |   26.7   |   61.0   |   28.8   |            |
|   QD-DETR   |   52.0   |   31.7   |   63.6   |   29.4   |            |
|     EaTR    |   48.4   |   27.5   |   59.9   |   26.9   |            |
|    UVCOM    |   48.4   |   27.1   |   60.9   |   27.9   |            |
|   CG-DETR   | **54.4** | **31.8** | **65.5** | **30.5** |            |

#### CLIP+Slowfast
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   53.4   |   30.7   |   62.0   |   29.1   |            |
|   QD-DETR   | **59.4** | **37.9** | **66.6** | **33.8** |            |
|     EaTR    |   55.2   |   33.1   |   65.4   |   30.4   |            |
|    UVCOM    |   56.9   |   35.9   |   65.6   |   33.6   |            |
|   CG-DETR   |   57.6   |   35.1   |   65.9   |   30.9   |            |

#### TaCoS (Moment retrieval)
#### ResNet152+GloVe
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   20.0   |    8.6   |   24.2   |    6.9   |            |
|   QD-DETR   |   30.6   |   15.1   |   35.1   |   12.3   |            |
|     EaTR    |   22.5   |    9.2   |   26.3   |    7.9   |            |
|    UVCOM    |   24.1   |   10.7   |   28.1   |    8.6   |            |
|   CG-DETR   | **34.2** | **17.4** | **39.7** | **14.6** |            |

#### CLIP
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   18.0   |    7.9   |   21.3   |    6.7   |            |
|   QD-DETR   |   32.3   |   17.2   |   36.0   |   14.1   |            |
|     EaTR    |   24.7   |   10.0   |   28.8   |    8.7   |            |
|    UVCOM    | **36.8** | **20.0** | **41.5** | **16.3** |            |
|   CG-DETR   |   34.3   |   19.8   |   38.6   |   15.8   |            |

#### CLIP+Slowfast
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   25.5   |   12.9   |   29.1   |   10.3   |            |
|   QD-DETR   |   38.7   |   22.1   |   42.9   |   16.7   |            |
|     EaTR    |   31.7   |   15.6   |   37.4   |   14.0   |            |
|    UVCOM    |   40.2   |   23.3   |   43.5   |   19.1   |            |
|   CG-DETR   | **39.8** | **25.1** | **44.2** | **19.6** |            |

#### TVSum (Highlight detection)
#### ResNet152+GloVe
|    Models   |    mAP   | checkpoint |
|:-----------:|:--------:|:----------:|
| Moment DETR |   85.9   |            |
|   QD-DETR   |   87.2   |            |
|     EaTR    |   86.2   |            |
|    UVCOM    | **87.6** |            |
|   CG-DETR   |   87.1   |            |

#### CLIP
|    Models   |    mAP   | checkpoint |
|:-----------:|:--------:|:----------:|
| Moment DETR | **89.1** |            |
|   QD-DETR   |   88.4   |            |
|     EaTR    |   86.7   |            |
|    UVCOM    |   87.7   |            |
|   CG-DETR   |   88.1   |            |

#### CLIP+Slowfast
|    Models   |    mAP   | checkpoint |
|:-----------:|:--------:|:----------:|
| Moment DETR |   86.9   |            |
|   QD-DETR   |   88.4   |            |
|     EaTR    |   86.1   |            |
|    UVCOM    |   87.7   |            |
|   CG-DETR   | **88.8** |            |

#### I3D+CLIP (Text)
|    Models   |    mAP   | checkpoint |
|:-----------:|:--------:|:----------:|
| Moment DETR |   86.7   |            |
|   QD-DETR   |   87.1   |            |
|     EaTR    |   85.0   |            |
|    UVCOM    | **87.9** |            |
|   CG-DETR   | **88.9** |            |

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## LICENSE
Apache License 2.0

## Contact
Taichi Nishimura ([taichitary@gmail.com](taichitary@gmail.com))
