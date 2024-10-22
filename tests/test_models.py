import os
import math
import pytest
import subprocess
import random
from lighthouse.models import (MomentDETRPredictor, QDDETRPredictor, EaTRPredictor, 
                               CGDETRPredictor, UVCOMPredictor)


FEATURES = ['clip', 'clip_slowfast']
AUDIO_FEATURES = ['clap']
MODELS = ['moment_detr', 'qd_detr', 'eatr', 'cg_detr', 'uvcom']
DATASETS = ['qvhighlight']
MIN_DURATION = 10
MAX_DURATION = 151
MIN_DURATION_AUDIO = 10
MAX_DURATION_AUDIO = 50
MOMENT_NUM = 10
SAMPLE_NUM = 10


@pytest.mark.dependency()
def test_generate_multiple_duration_videos():
    durations = [i for i in range(MIN_DURATION, MAX_DURATION)]
    return_codes = []
    for duration in durations:
        cmd = f'ffmpeg -y -i api_example/RoripwjYFp8_60.0_210.0.mp4 -t {duration} -c copy tests/test_videos/video_duration_{duration}.mp4'
        result = subprocess.run(cmd, shell=True)
        return_codes.append(result.returncode)
    for return_code in return_codes:
        assert return_code == 0, '[ffmpeg conversion] return_code should be set 0.'

@pytest.mark.dependency()
def test_generate_multiple_duration_audios():
    durations = [i for i in range(MIN_DURATION_AUDIO, MAX_DURATION_AUDIO)]
    return_codes = []
    for duration in durations:
        cmd = f'ffmpeg -y -i api_example/1a-ODBWMUAE.wav -t {duration} -c copy tests/test_audios/audio_duration_{duration}.wav'
        result = subprocess.run(cmd, shell=True)
        return_codes.append(result.returncode)
    for return_code in return_codes:
        assert return_code == 0, '[ffmpeg conversion] return_code should be set 0.'

@pytest.mark.dependency()
def test_save_model_weights():
    return_codes = []
    for feature in FEATURES:
        for model in MODELS:
            for dataset in DATASETS:
                if not os.path.exists(f'tests/weights/{feature}_{model}_{dataset}.ckpt'):
                    cmd = f'wget -P tests/weights/ https://zenodo.org/records/13363606/files/{feature}_{model}_{dataset}.ckpt'
                    result = subprocess.run(cmd, shell=True)
                    return_codes.append(result.returncode)

    for feature in AUDIO_FEATURES:
        for model in ['qd_detr']:
            for dataset in ['clotho-moment']:
                if not os.path.exists(f'tests/weights/{feature}_{model}_{dataset}.ckpt'):
                    cmd = f'wget -P tests/weights/ https://zenodo.org/records/13961029/files/{feature}_{model}_{dataset}.ckpt'
                    result = subprocess.run(cmd, shell=True)
                    return_codes.append(result.returncode)
    for return_code in return_codes:
        assert return_code == 0, '[save model weights] return_code should be set 0.'

@pytest.mark.dependency()
def test_load_slowfast_pann_weights():
    if not os.path.exists('tests/SLOWFAST_8x8_R50.pkl'):
        result = subprocess.run('wget -P tests/ https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl', shell=True)
        assert result.returncode == 0, '[Save slowfast weights] return_code should be set 0.'
    
    if not os.path.exists('tests/Cnn14_mAP=0.431.pth'):
        result = subprocess.run('wget -P tests/ https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth', shell=True)
        assert result.returncode == 0, '[Save PANNs weights] return_code should be set 0.'

@pytest.mark.dependency(depends=['test_generate_multiple_duration_videos', 
                                 'test_save_model_weights', 
                                 'test_load_slowfast_pann_weights'])
def test_video_model_prediction():
    """
    Test all of the trained video moment retrieval models, except for resnet_glove features and taskweave
    Untested features:
        - ResNet+GloVe is skipped due to their low performance.
        - CLIP+Slowfast+PANNs is skipped due to their low latency.
    
    Untested models:
        - TR-DETR is skipped because model use .cuda() function. We need to remove it.
        - Taskweave is skiped because two strategies are neccesary for prediction.
    """
    model_loaders  = {
        'moment_detr': MomentDETRPredictor,
        'qd_detr': QDDETRPredictor,
        'eatr': EaTRPredictor,
        'cg_detr': CGDETRPredictor,
        'uvcom': UVCOMPredictor,
    }

    # test video features
    for feature in FEATURES:
        for model_name in MODELS:
            for dataset in DATASETS:
                model_weight = os.path.join('tests/weights/', f'{feature}_{model_name}_{dataset}.ckpt')
                model = model_loaders[model_name](model_weight, device='cpu', feature_name=feature, 
                                                slowfast_path='tests/SLOWFAST_8x8_R50.pkl', 
                                                pann_path='tests/Cnn14_mAP=0.431.pth')
                
                # test 10 duration samples
                durations = random.sample([i for i in range(MIN_DURATION, MAX_DURATION)], SAMPLE_NUM)
                for second in durations:
                    video_path = f'tests/test_videos/video_duration_{second}.mp4'
                    model.encode_video(video_path)
                    query = 'A woman wearing a glass is speaking in front of the camera'
                    prediction = model.predict(query)
                    assert len(prediction['pred_relevant_windows']) == MOMENT_NUM, \
                        f'The number of moments from {feature}_{model_name}_{dataset} is expected {MOMENT_NUM}, but got {len(prediction["pred_relevant_windows"])}.'
                    assert len(prediction['pred_saliency_scores']) == math.ceil(second / model._clip_len), \
                        f'The number of saliency scores from {feature}_{model_name}_{dataset} is expected {math.ceil(second / model._clip_len)}, but got {len(prediction["pred_saliency_scores"])}.'

@pytest.mark.dependency(depends=['test_generate_multiple_duration_audios', 
                                 'test_save_model_weights'])
def test_audio_model_prediction():
    """
    Test all of the trained audio moment retrieval models (now only supports QD-DETR with CLAP features.)
    """
    # test audio features
    for feature in AUDIO_FEATURES:
        for model_name in ['qd_detr']:
            for dataset in ['clotho-moment']:
                model_weight = os.path.join('tests/weights/', f'{feature}_{model_name}_{dataset}.ckpt')
                model = QDDETRPredictor(model_weight, device='cpu', feature_name=feature, 
                                        slowfast_path=None, 
                                        pann_path=None)
                
                # test 10 duration samples
                durations = random.sample([i for i in range(MIN_DURATION_AUDIO, MAX_DURATION_AUDIO)], SAMPLE_NUM)
                for second in durations:
                    audio_path = f'tests/test_audios/audio_duration_{second}.wav'
                    model.encode_audio(audio_path)
                    query = 'Water cascades down from a waterfall.'
                    prediction = model.predict(query)
                    assert len(prediction['pred_relevant_windows']) == MOMENT_NUM, \
                        f'The number of moments from {feature}_{model_name}_{dataset} is expected {MOMENT_NUM}, but got {len(prediction["pred_relevant_windows"])}.'
                    assert len(prediction['pred_saliency_scores']) == math.ceil(second / model._clip_len), \
                        f'The number of saliency scores from {feature}_{model_name}_{dataset} is expected {math.ceil(second / model._clip_len)}, but got {len(prediction["pred_saliency_scores"])}.'
