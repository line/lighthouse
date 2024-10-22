from setuptools import setup, find_packages

setup(
    name='lighthouse',
    version='0.1',
    install_requires=['easydict', 'pandas', 'tqdm', 'pyyaml', 'scikit-learn', 'ffmpeg-python',
                      'ftfy', 'regex', 'einops', 'fvcore', 'gradio', 'torchlibrosa', 'librosa',
                      'msclap', 'clip@git+https://github.com/openai/CLIP.git'],
    packages=find_packages(exclude=['training']),
)
