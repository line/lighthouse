import subprocess

def generate_multiple_duration_videos():
    MIN_DURATION = 10
    MAX_DURATION = 151
    durations = [i for i in range(MIN_DURATION, MAX_DURATION)]
    for duration in durations:
        cmd = 'ffmpeg -i api_example/RoripwjYFp8_60.0_210.0.mp4 -t {} -c copy tests/test_videos/video_duration_{}.mp4'.format(duration, duration)
        subprocess.run(cmd, shell=True)    
    return True


def test_model_prediction():
    features = ['resnet_glove', 'clip', 'clip_slowfast', 'clip_slowfast_pann']
    models = ['moment_detr', 'qd_detr', 'eatr', 'cg_detr', 'uvcom', 'tr_detr', 'taskweave_mr2hd', 'taskweave_hd2mr']
    datasets = ['qvhighlight']
    # TODO: test all of the models on all of the settings.
    return True