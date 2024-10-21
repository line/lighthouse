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
import ffmpeg
import pandas as pd
import gradio as gr
from tqdm import tqdm
from lighthouse.models import *

# use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAMES = ['cg_detr', 'moment_detr', 'eatr', 'qd_detr', 'tr_detr', 'uvcom']
FEATURES = ['clip', 'clip_slowfast', 'clip_slowfast_pann']
TOPK_MOMENT = 5
TOPK_HIGHLIGHT = 5

"""
Helper functions
"""
def load_pretrained_weights():
    file_urls = []
    for model_name in MODEL_NAMES:
        for feature in FEATURES:
            file_urls.append(
                "https://zenodo.org/records/13960580/files/{}_{}_qvhighlight.ckpt".format(feature, model_name)
            )
    for file_url in tqdm(file_urls):
        if not os.path.exists('gradio_demo/weights/' + os.path.basename(file_url)):
            command = 'wget -P gradio_demo/weights/ {}'.format(file_url)
            subprocess.run(command, shell=True)

    # Slowfast weights
    if not os.path.exists('SLOWFAST_8x8_R50.pkl'):
        subprocess.run('wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl', shell=True)

    # PANNs weights
    if not os.path.exists('Cnn14_mAP=0.431.pth'):
        subprocess.run('wget https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth', shell=True)

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
model = CGDETRPredictor('gradio_demo/weights/clip_cg_detr_qvhighlight.ckpt', device=device, 
                        feature_name='clip', slowfast_path=None, pann_path=None)

js_codes = ["""() => {{
            let moment_text = document.getElementById('result_{}').textContent;
            var replaced_text = moment_text.replace(/moment..../, '').replace(/\ Score.*/, '');
            let start_end = JSON.parse(replaced_text);
            document.getElementsByTagName("video")[0].currentTime = start_end[0];
            document.getElementsByTagName("video")[0].play();
        }}""".format(i) for i in range(TOPK_MOMENT)]

"""
Gradio functions
"""
def video_upload(video):
    if video is None:
        model.video_feats = None
        model.video_mask = None
        model.video_path = None
        yield gr.update(value="Removed the video", visible=True)
    else:
        yield gr.update(value="Processing the video. Wait for a minute...", visible=True)
        model.encode_video(video)
        yield gr.update(value="Finished video processing!", visible=True)

def model_load(radio):
    if radio is not None:
        yield gr.update(value="Loading new model. Wait for a minute...", visible=True)
        global model
        feature, model_name = radio.split('+')
        feature, model_name = feature.strip(), model_name.strip()

        if model_name == 'moment_detr':
            model_class = MomentDETRPredictor
        elif model_name == 'qd_detr':
            model_class = QDDETRPredictor
        elif model_name == 'eatr':
            model_class = EaTRPredictor
        elif model_name == 'tr_detr':
            model_class = TRDETRPredictor
        elif model_name == 'uvcom':
            model_class = UVCOMPredictor
        elif model_name == 'cg_detr':
            model_class = CGDETRPredictor
        else:
            raise gr.Error("Select from the models")
        
        model = model_class('gradio_demo/weights/{}_{}_qvhighlight.ckpt'.format(feature, model_name),
                            device=device, feature_name='{}'.format(feature),
                            slowfast_path='SLOWFAST_8x8_R50.pkl', pann_path='Cnn14_mAP=0.431.pth')
        yield gr.update(value="Model loaded: {}".format(radio), visible=True)

def predict(textbox, line, gallery):
    prediction = model.predict(textbox)
    if prediction is None:
        raise gr.Error('Upload the video before pushing the `Retrieve moment & highlight detection` button.')
    else:
        mr_results = prediction['pred_relevant_windows']
        hl_results = prediction['pred_saliency_scores']

        buttons = []
        for i, pred in enumerate(mr_results[:TOPK_MOMENT]):
            buttons.append(gr.Button(value='moment {}: [{}, {}] Score: {}'.format(i+1, pred[0], pred[1], pred[2]), visible=True))
        
        # Visualize the HD score
        seconds = [model._vision_encoder._clip_len * i for i in range(len(hl_results))]
        hl_data = pd.DataFrame({ 'second': seconds, 'saliency_score': hl_results })
        min_val, max_val = min(hl_results), max(hl_results) + 1
        min_x, max_x = min(seconds), max(seconds)
        line = gr.LinePlot(value=hl_data, x='second', y='saliency_score', visible=True, y_lim=[min_val, max_val], x_lim=[min_x, max_x])

        # Show highlight frames
        n_largest_df = hl_data.nlargest(columns='saliency_score', n=TOPK_HIGHLIGHT)
        highlighted_seconds = n_largest_df.second.tolist()
        highlighted_scores = n_largest_df.saliency_score.tolist()

        output_image_paths = []
        for i, (second, score) in enumerate(zip(highlighted_seconds, highlighted_scores)):
            output_path = "gradio_demo/highlight_frames/highlight_{}.png".format(i)
            (
                ffmpeg
                .input(model._video_path, ss=second)
                .output(output_path, vframes=1, qscale=2)
                .global_args('-loglevel', 'quiet', '-y')
                .run()
            )
            output_image_paths.append((output_path, "Highlight: {} - score: {:.02f}".format(i+1, score)))
        gallery = gr.Gallery(value=output_image_paths, label='gradio', columns=5, show_download_button=True, visible=True)
        return buttons + [line, gallery]


def main():
    title = """# Moment Retrieval & Highlight Detection Demo"""
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(title)

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("## Model selection")
                    radio_list = flatten([["{} + {}".format(feature, model_name) for model_name in MODEL_NAMES] for feature in FEATURES])
                    radio = gr.Radio(radio_list, label="models", value="clip + cg_detr", info="Which model do you want to use?")
                    load_status_text = gr.Textbox(label='Model load status', value='Model loaded: clip + cg_detr')

                with gr.Group():
                    gr.Markdown("## Video and query")
                    video_input = gr.Video(elem_id='video', height=600)
                    output = gr.Textbox(label='Video processing progress')
                    query_input = gr.Textbox(label='query')
                    button = gr.Button("Retrieve moment & highlight detection", variant="primary")
            
            with gr.Column():
                with gr.Group():
                    gr.Markdown("## Retrieved moments")

                    button_1 = gr.Button(value='moment 1', visible=False, elem_id='result_0')
                    button_2 = gr.Button(value='moment 2', visible=False, elem_id='result_1')
                    button_3 = gr.Button(value='moment 3', visible=False, elem_id='result_2')
                    button_4 = gr.Button(value='moment 4', visible=False, elem_id='result_3')
                    button_5 = gr.Button(value='moment 5', visible=False, elem_id='result_4')

                    button_1.click(None, None, None, js=js_codes[0])
                    button_2.click(None, None, None, js=js_codes[1])
                    button_3.click(None, None, None, js=js_codes[2])
                    button_4.click(None, None, None, js=js_codes[3])
                    button_5.click(None, None, None, js=js_codes[4])

                # dummy
                with gr.Group():
                    gr.Markdown("## Saliency score")
                    line = gr.LinePlot(value=pd.DataFrame({'x': [], 'y': []}), x='x', y='y', visible=False)
                    gr.Markdown("### Highlighted frames")
                    gallery = gr.Gallery(value=[], label="highlight", columns=5, visible=False)
                
                video_input.change(video_upload, inputs=[video_input], outputs=output)
                radio.select(model_load, inputs=[radio], outputs=load_status_text)
                
                button.click(predict, 
                            inputs=[query_input, line, gallery], 
                            outputs=[button_1, button_2, button_3, button_4, button_5, line, gallery])

    demo.launch()

if __name__ == "__main__":
    main()