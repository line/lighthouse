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
import torch
import subprocess
import gradio as gr
import librosa
from tqdm import tqdm
from lighthouse.models import *

# use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAMES = ['qd_detr']
FEATURES = ['clap']
TOPK_MOMENT = 5

"""
Helper functions
"""
def load_pretrained_weights():
    file_urls = []
    for model_name in MODEL_NAMES:
        for feature in FEATURES:
            file_urls.append(
                "https://zenodo.org/records/13961029/files/{}_{}_clotho-moment.ckpt".format(feature, model_name)
            )
    for file_url in tqdm(file_urls):
        if not os.path.exists('gradio_demo/weights/' + os.path.basename(file_url)):
            command = 'wget -P gradio_demo/weights/ {}'.format(file_url)
            subprocess.run(command, shell=True)

    return file_urls

def flatten(array2d):
    list1d = []
    for elem in array2d:
        list1d += elem
    return list1d

"""
Model initialization
"""
load_pretrained_weights()
model = QDDETRPredictor('gradio_demo/weights/clap_qd_detr_clotho-moment.ckpt', device=device, feature_name='clap')

"""
Gradio functions
"""
def audio_upload(audio):
    if audio is None:
        model.audio_feats = None
        yield gr.update(value="Removed the audio", visible=True)
    else:
        yield gr.update(value="Processing the audio. Wait for a minute...", visible=True)
        model.encode_audio(audio)
        yield gr.update(value="Finished audio processing!", visible=True)

def model_load(radio):
    if radio is not None:
        yield gr.update(value="Loading new model. Wait for a minute...", visible=True)
        global model
        feature, model_name = radio.split('+')
        feature, model_name = feature.strip(), model_name.strip()

        if model_name == 'qd_detr':
            model_class = QDDETRPredictor
        else:
            raise gr.Error("Select from the models")

        model = model_class('gradio_demo/weights/{}_{}_clotho-moment.ckpt'.format(feature, model_name),
                            device=device, feature_name='{}'.format(feature))
        yield gr.update(value="Model loaded: {}".format(radio), visible=True)

def predict(textbox, line, gallery):
    prediction = model.predict(textbox)
    if prediction is None:
        raise gr.Error('Upload the audio before pushing the `Retrieve moment` button.')
    else:
        mr_results = prediction['pred_relevant_windows']

        buttons = []
        for i, pred in enumerate(mr_results[:TOPK_MOMENT]):
            buttons.append(gr.Button(value='moment {}: [{}, {}] Score: {}'.format(i+1, pred[0], pred[1], pred[2]), visible=True))

        return buttons


def show_trimmed_audio(audio, button):
    s, sr = librosa.load(audio, sr=None)
    _seconds = button.split(': [')[1].split(']')[0].split(', ')
    start_sec = float(_seconds[0])
    end_sec = float(_seconds[1])
    start_frame = int(start_sec * sr)
    end_frame = int(end_sec * sr)

    return gr.Audio((sr, s[start_frame:end_frame]), interactive=False, visible=True)


def main():
    title = """# Audio Moment Retrieval Demo"""

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(title)

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("## Model selection")
                    radio_list = flatten([["{} + {}".format(feature, model_name) for model_name in MODEL_NAMES] for feature in FEATURES])
                    radio = gr.Radio(radio_list, label="models", value="clap + qd_detr", info="Which model do you want to use?")
                    load_status_text = gr.Textbox(label='Model load status', value='Model loaded: clap + qd_detr')

                with gr.Group():
                    gr.Markdown("## Audio and query")
                    audio_input = gr.Audio(type='filepath')
                    output = gr.Textbox(label='Audio processing progress')
                    query_input = gr.Textbox(label='query')
                    button = gr.Button("Retrieve moment", variant="primary")

            with gr.Column():
                with gr.Group():
                    gr.Markdown("## Retrieved moments")

                    button_1 = gr.Button(value='moment 1', visible=False, elem_id='result_0')
                    button_2 = gr.Button(value='moment 2', visible=False, elem_id='result_1')
                    button_3 = gr.Button(value='moment 3', visible=False, elem_id='result_2')
                    button_4 = gr.Button(value='moment 4', visible=False, elem_id='result_3')
                    button_5 = gr.Button(value='moment 5', visible=False, elem_id='result_4')
                    result = gr.Audio(None, label='Trimmed audio', interactive=False, visible=False)

                    button_1.click(show_trimmed_audio, inputs=[audio_input, button_1], outputs=[result])
                    button_2.click(show_trimmed_audio, inputs=[audio_input, button_2], outputs=[result])
                    button_3.click(show_trimmed_audio, inputs=[audio_input, button_3], outputs=[result])
                    button_4.click(show_trimmed_audio, inputs=[audio_input, button_4], outputs=[result])
                    button_5.click(show_trimmed_audio, inputs=[audio_input, button_5], outputs=[result])

                audio_input.change(audio_upload, inputs=[audio_input], outputs=output)
                radio.select(model_load, inputs=[radio], outputs=load_status_text)
                
                button.click(predict, 
                            inputs=[query_input], 
                            outputs=[button_1, button_2, button_3, button_4, button_5])

    demo.launch()

if __name__ == "__main__":
    main()
