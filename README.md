# Lighthouse

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
or
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
Download [pre-trained weights](https://drive.google.com/file/d/1ebQbhH1tjgTmRBmyOoW8J9DH7s80fqR9/view?usp=drive_link) and Run `python api_example/demo.py` to reproduce the results. In addition, to use `clip_slowfast` features, it is necessary to download slowfast pre-trained weights ([SLOWFAST_8x8_R50](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl)).

**Limitation**: The maximum video duration is **150s** due to the current benchmark datasets. Using CLIP+Slowfast feature (`feature_name=clip_slowfast`) is possible for CPU users, but very slow. Use `feature_name='clip'`.

## Gradio demo
Run `python gradio_demo/demo.py`. Upload the video and input text query, and click the blue button.

![Gradio demo image](images/demo_improved.png)

## Supported models, datasets, and features
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

### Training and evaluation
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

## Reproduced results

### QVHighlights (Moment retrieval & highlight detection)
Test set scores are reported.

#### ResNet152+GloVe
|    Models   |  R1@0.5  |  R1@0.7  |  HD mAP  |   HIT@1  | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   40.0   |   22.0   |   30.0   |   42.9   |[ckpt](https://drive.google.com/file/d/1kadbbHie1dk2G6uts-daTnuu2guNIUCk/view?usp=sharing)|
|   QD-DETR   |   52.7   |   36.1   |   33.8   |   50.7   |[ckpt](https://drive.google.com/file/d/1aC3ghUQMLAyEi1fNrbMvO7osKd2_VShB/view?usp=sharing)|
|     EaTR    | **57.2** | **38.9** | **36.3** | **57.4** |[ckpt](https://drive.google.com/file/d/1_n22fn-sZy-YM5eAE0mJVbOrfS_FPwm0/view?usp=sharing)|
|   TR-DETR   |   47.7   |   31.6   |   34.3   |   52.0   |[ckpt](https://drive.google.com/file/d/1Z_NyefuQMmLzkTtVpSLwdkdsqTItdBOR/view?usp=sharing)|
|    UVCOM    |   53.8   |   37.6   |   34.8   |   53.8   |[ckpt](https://drive.google.com/file/d/1UY51b4TixPaJUHpr6UU7F1M-sS2l4aY_/view?usp=sharing)|
|   CG-DETR   |   53.1   |   38.3   |   34.5   |   52.9   |[ckpt](https://drive.google.com/file/d/1Mk3lZM7qkzcYCKQBVgwXv3B5Z5KHMtwV/view?usp=sharing)|

#### CLIP
|    Models   |  R1@0.5  |  R1@0.7  |  HD mAP  |   HIT@1  | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   55.8   |   33.8   |   35.7   |   55.8   |[ckpt](https://drive.google.com/file/d/1j4w0T_R34FP8u5jjaOFOxwCUYlxZgByZ/view?usp=sharing)|
|   QD-DETR   |   60.8   |   41.8   |   38.2   |   60.7   |[ckpt](https://drive.google.com/file/d/1nKZxLGmIckIcbElknbBseG0JWfwp8a4V/view?usp=sharing)|
|     EaTR    |   54.6   |   34.0   |   34.9   |   54.7   |[ckpt](https://drive.google.com/file/d/1fK6ZZXEaAAemJZP0J1BBqfOkgy1Y_XGs/view?usp=sharing)|
|   TR-DETR   |   60.2   |   41.4   |   38.6   |   59.3   |[ckpt](https://drive.google.com/file/d/1Z71dOFq9tFhuAQPOkTv4HK8FufUjZplJ/view?usp=sharing)|
|    UVCOM    |   62.7   | **46.9** | **39.8** | **64.5** |[ckpt](https://drive.google.com/file/d/1i8g5vakVkPO93oQBFIFP7GNi7Ru95MAn/view?usp=sharing)|
|   CG-DETR   | **64.5** |   46.0   |   39.4   |   64.3   |[ckpt](https://drive.google.com/file/d/1uzyba_mPPe73lnZ7w51IUIjL2Fglgg0a/view?usp=sharing)|

#### CLIP+Slowfast
|    Models   |  R1@0.5  |  R1@0.7  |  HD mAP  |   HIT@1  | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   54.4   |   33.9   |   32.6   |   56.7   |[ckpt](https://drive.google.com/file/d/1GXjhcJz_XvbsTOuWlJI8OcX_wRnd-QYk/view?usp=sharing)|
|   QD-DETR   |   62.1   |   44.6   |   38.8   |   61.6   |[ckpt](https://drive.google.com/file/d/1F2alwuiAtWyzbyPF25RnUZhaiTSmNJYJ/view?usp=sharing)|
|     EaTR    |   57.2   |   38.9   |   36.6   |   57.9   |[ckpt](https://drive.google.com/file/d/1h7JnU2CbB1el9kSdaZFSdgmTjGZOqClO/view?usp=sharing)|
|   TR-DETR   | **65.2** | **48.8** |   39.8   |   62.1   |[ckpt](https://drive.google.com/file/d/11vLZgEQUyAZ2DG9LGDKJ8jCXR9kUMJh9/view?usp=sharing)|
|    UVCOM    |   62.6   |   47.6   |   39.6   |   62.8   |[ckpt](https://drive.google.com/file/d/1dVuR6FMWcayrxZMfsBVBr-RxThF48_47/view?usp=sharing)|
|   CG-DETR   |   64.9   |   48.1   | **40.7** | **67.0** |[ckpt](https://drive.google.com/file/d/1E0Jnf10dCUTV9jCth9YpfjwT4Bw_5w0i/view?usp=sharing)|

### ActivityNet Captions (Moment retrieval)
Val_1 scores are reported.

#### ResNet152+GloVe
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   34.2   |   19.5   |   46.3   |   24.4   |[ckpt](https://drive.google.com/file/d/1IfW9LFEuTmFV-GN5G9meLkNmyzeRD2JW/view?usp=sharing)|
|   QD-DETR   |   35.4   |   20.3   |   47.4   |   24.9   |[ckpt](https://drive.google.com/file/d/1MrxWOt3hE_9TdmQIl8UqeWqFjSkirVRL/view?usp=sharing)|
|     EaTR    |   32.4   |   18.2   |   44.3   |   21.9   |[ckpt](https://drive.google.com/file/d/1iEUMfYKqK6GEexcshhYWcRy58UXA89zw/view?usp=sharing)|
|    UVCOM    |   34.4   |   19.9   |   46.1   |   24.4   |[ckpt](https://drive.google.com/file/d/18ediEySQ8qdBns1I9j2IUqoWOn9g6fMJ/view?usp=sharing)|
|   CG-DETR   | **37.0** | **21.2** | **48.6** | **26.5** |[ckpt](https://drive.google.com/file/d/1WmKWfLcjJCVaWUm2M3Q-oJmTSHkrswLi/view?usp=sharing)|

#### CLIP
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   36.1   |   20.4   |   48.2   |   25.7   |[ckpt](https://drive.google.com/file/d/1ucJEg9r5-dFov-aTd_BbBBVwi6_RKFzO/view?usp=sharing)|
|   QD-DETR   |   36.9   |   21.4   |   48.4   |   26.3   |[ckpt](https://drive.google.com/file/d/1qJgbVm2jYXrGXYD4poexXkMOe4Cy6j2u/view?usp=sharing)|
|     EaTR    |   34.6   |   19.7   |   45.1   |   23.1   |[ckpt](https://drive.google.com/file/d/1DDpk1V4t4GswMLc-CArQUXRAnZiUJY-p/view?usp=sharing)|
|    UVCOM    |   37.0   |   21.5   |   48.3   |   25.7   |[ckpt](https://drive.google.com/file/d/1n7aJqIXyPs14__m3bHE_AsjErYCBLh62/view?usp=sharing)|
|   CG-DETR   | **38.8** | **22.6** | **50.6** | **27.5** |[ckpt](https://drive.google.com/file/d/1cdjkgUoCN5DTlRQ0xzV7CMhCveLP_G6L/view?usp=sharing)|

#### CLIP+Slowfast
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   36.5   |   21.1   |   48.4   |   26.0   |[ckpt](https://drive.google.com/file/d/13Z697Bd1B0-2EBUDSUi5tNodzjKqL9D8/view?usp=sharing)|
|   QD-DETR   |   37.5   |   22.1   |   48.9   |   26.4   |[ckpt](https://drive.google.com/file/d/1tNMobXj1w4u0ZLWubhUcWDS8dYBh339y/view?usp=sharing)|
|     EaTR    |   34.6   |   19.3   |   45.2   |   22.3   |[ckpt](https://drive.google.com/file/d/1nAFCKQyXUT02HXIbCSrE4RuUGoDfIaeh/view?usp=sharing)|
|    UVCOM    |   37.3   |   21.6   |   48.9   |   25.7   |[ckpt](https://drive.google.com/file/d/1gQnC8LSNaLJiJNKl0UZRyRU2tKqb1EyD/view?usp=sharing)|
|   CG-DETR   | **40.0** | **23.2** | **51.0** | **27.7** |[ckpt](https://drive.google.com/file/d/1DUyjSKLdWf4O8lVEDQUPLenP0UJ9Okno/view?usp=sharing)|

### Charades-STA (Moment retrieval)
Test set scores are reported.

#### ResNet152+GloVe
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   38.4   |   22.9   |   52.4   |   22.2   |[ckpt](https://drive.google.com/file/d/1f8qVioF5G6a-4uFVteog24Mp0BQfGTi9/view?usp=sharing)|
|   QD-DETR   | **42.1** | **24.0** |   56.7   | **24.5** |[ckpt](https://drive.google.com/file/d/1vf2bczWP3RukASjnSsFRBa8DIDooc3vW/view?usp=sharing)|
|     EaTR    |   37.6   |   20.1   |   53.5   |   23.6   |[ckpt](https://drive.google.com/file/d/1df8rzL9agkGnEIchCKR1KApFn4WqzMf9/view?usp=sharing)|
|    UVCOM    |   38.1   |   18.2   |   54.4   |   21.1   |[ckpt](https://drive.google.com/file/d/1-cqMH6MYOPEe73Uj4Edwdvc2rBC_Ay9H/view?usp=sharing)|
|   CG-DETR   |   39.7   |   19.4   | **56.9** |   23.2   |[ckpt](https://drive.google.com/file/d/16Ml3Sh2R1lYDaQteREntJ-7_TtnJy9rO/view?usp=sharing)|

#### CLIP
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   47.9   |   26.7   |   61.0   |   28.8   |[ckpt](https://drive.google.com/file/d/1OdPLIAs7eRUesCOqs32MeCrB9V8q7amK/view?usp=sharing)|
|   QD-DETR   |   52.0   |   31.7   |   63.6   |   29.4   |[ckpt](https://drive.google.com/file/d/1bqq70DwvvK2CTnT59_YvjXMP1APQKrH2/view?usp=sharing)|
|     EaTR    |   48.4   |   27.5   |   59.9   |   26.9   |[ckpt](https://drive.google.com/file/d/1XUCiyiF-dFg0CSkZCsRfiyZGjpiEHLWh/view?usp=sharing)|
|    UVCOM    |   48.4   |   27.1   |   60.9   |   27.9   |[ckpt](https://drive.google.com/file/d/1YiR3B73E_OHdKbqkISQiebvu8c_IG0EF/view?usp=sharing)|
|   CG-DETR   | **54.4** | **31.8** | **65.5** | **30.5** |[ckpt](https://drive.google.com/file/d/137E506J9pkud0kaOtUKy38qvXkxySUjf/view?usp=sharing)|

#### CLIP+Slowfast
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   53.4   |   30.7   |   62.0   |   29.1   |[ckpt](https://drive.google.com/file/d/1ih1SgTct1KFYHWAGqIU5OovcKJtPUaq7/view?usp=sharing)|
|   QD-DETR   | **59.4** | **37.9** | **66.6** | **33.8** |[ckpt](https://drive.google.com/file/d/1sWLvUjQ8gtWTyIYasemsTE_X9Dp3nK5A/view?usp=sharing)|
|     EaTR    |   55.2   |   33.1   |   65.4   |   30.4   |[ckpt](https://drive.google.com/file/d/1ZEmJu8hML4FnPpHLIPNs58HWO10sbMcp/view?usp=sharing)|
|    UVCOM    |   56.9   |   35.9   |   65.6   |   33.6   |[ckpt](https://drive.google.com/file/d/1-sOks0Y69Pla0Q4Yo0wzjiJVN77SxVSX/view?usp=sharing)|
|   CG-DETR   |   57.6   |   35.1   |   65.9   |   30.9   |[ckpt](https://drive.google.com/file/d/1_3cQXwZcS8sVnNFrRVslW1BnmoZ3bDaI/view?usp=sharing)|

### TaCoS (Moment retrieval)
Test set scores are reported.

#### ResNet152+GloVe
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   20.0   |    8.6   |   24.2   |    6.9   |[ckpt](https://drive.google.com/file/d/1jS5qHfoKIjDEvlrgCMFlQEJU9NR1nygT/view?usp=sharing)|
|   QD-DETR   |   30.6   |   15.1   |   35.1   |   12.3   |[ckpt](https://drive.google.com/file/d/17BQZ9EtC8f0Nny-dkZ8HVeJpFfalJOEE/view?usp=sharing)|
|     EaTR    |   22.5   |    9.2   |   26.3   |    7.9   |[ckpt](https://drive.google.com/file/d/1XtTboZQMk_K5HDtjdYlJYWfKesqQtrwc/view?usp=sharing)|
|    UVCOM    |   24.1   |   10.7   |   28.1   |    8.6   |[ckpt](https://drive.google.com/file/d/1Ik8KU8nurKzWxNUuwQZMewJCCwwp54xR/view?usp=sharing)|
|   CG-DETR   | **34.2** | **17.4** | **39.7** | **14.6** |[ckpt](https://drive.google.com/file/d/1WgCTV-5sBcbpDu6KNVQ47ZOmjTJgVT0x/view?usp=sharing)|

#### CLIP
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   18.0   |    7.9   |   21.3   |    6.7   |[ckpt](https://drive.google.com/file/d/1jUxd9whO9n1YwOE_MZxEx1JInkFzmph6/view?usp=sharing)|
|   QD-DETR   |   32.3   |   17.2   |   36.0   |   14.1   |[ckpt](https://drive.google.com/file/d/17gz5vMScglGC1dqxp1HSC1AFcAu47h80/view?usp=sharing)|
|     EaTR    |   24.7   |   10.0   |   28.8   |    8.7   |[ckpt](https://drive.google.com/file/d/16E-iJHKWoyfRPKxqRwF2pejW5mxz4dJd/view?usp=sharing)|
|    UVCOM    | **36.8** | **20.0** | **41.5** | **16.3** |[ckpt](https://drive.google.com/file/d/1HcQkxpDtZFKAhLXadi_ZPP6pOsqM01Lu/view?usp=sharing)|
|   CG-DETR   |   34.3   |   19.8   |   38.6   |   15.8   |[ckpt](https://drive.google.com/file/d/1G8qL_Y3szD35JFYOS_tMhPYj-TJs4ZPN/view?usp=sharing)|

#### CLIP+Slowfast
|    Models   |  R1@0.5  |  R1@0.7  |  mAP@0.5 | mAP@0.75 | checkpoint |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Moment DETR |   25.5   |   12.9   |   29.1   |   10.3   |[ckpt](https://drive.google.com/file/d/1ZKX5QIMbP93fR50KqwHYCTDFQarnGzLo/view?usp=sharing)|
|   QD-DETR   |   38.7   |   22.1   |   42.9   |   16.7   |[ckpt](https://drive.google.com/file/d/1sKWxLMzzOUHIpc35eNfATUQpd8dHVrO8/view?usp=sharing)|
|     EaTR    |   31.7   |   15.6   |   37.4   |   14.0   |[ckpt](https://drive.google.com/file/d/16wZt42Mpeg9BmPHOzgxrsuPqBvwaMuBF/view?usp=sharing)|
|    UVCOM    |   40.2   |   23.3   |   43.5   |   19.1   |[ckpt](https://drive.google.com/file/d/1rqMzVVMcIT24GgYgj7QPGE1py-qutiso/view?usp=sharing)|
|   CG-DETR   | **39.8** | **25.1** | **44.2** | **19.6** |[ckpt](https://drive.google.com/file/d/1c1R5N1bz2jgi_K8QYtof7q78Nsi18o-r/view?usp=sharing)|

Due to the file size, we do not distribute the weights of TVSum (Highlight detection) on Google drive.
Download from [here](https://drive.google.com/file/d/1ebQbhH1tjgTmRBmyOoW8J9DH7s80fqR9/view?usp=drive_link) or contact me via email.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Todo
- [ ] : Support tensorboard
- [ ] : Support Wandb
- [ ] : Support DeDiMo
- [ ] : Support TaskWeave
- [ ] : Support ASR-based pretraining on QVHighlights
- [ ] : Support TVSum Audio training

## LICENSE
Apache License 2.0

## Contact
Taichi Nishimura ([taichitary@gmail.com](taichitary@gmail.com))
