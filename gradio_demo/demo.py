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

import pandas as pd
import gradio as gr
from lighthouse.models import CGDETRPredictor

model = CGDETRPredictor('results/clip_cg_detr/qvhighlight/best.ckpt', 
                        device='cpu', feature_name='clip', slowfast_path='SLOWFAST_8x8_R50.pkl')

js_code_1 = """
() => {
    let moment_text = document.getElementById('result_1').textContent;
    var replaced_text = moment_text.replace(/moment..../, '').replace(/\ Score.*/, '');
    let start_end = JSON.parse(replaced_text);

    document.getElementsByTagName("video")[0].currentTime = start_end[0];
    document.getElementsByTagName("video")[0].play();
}
"""

js_code_2 = """
() => {
    let moment_text = document.getElementById('result_2').textContent;
    var replaced_text = moment_text.replace(/moment..../, '').replace(/\ Score.*/, '');
    let start_end = JSON.parse(replaced_text);

    document.getElementsByTagName("video")[0].currentTime = start_end[0];
    document.getElementsByTagName("video")[0].play();
}
"""

js_code_3 = """
() => {
    let moment_text = document.getElementById('result_3').textContent;
    var replaced_text = moment_text.replace(/moment..../, '').replace(/\ Score.*/, '');
    let start_end = JSON.parse(replaced_text);

    document.getElementsByTagName("video")[0].currentTime = start_end[0];
    document.getElementsByTagName("video")[0].play();
}
"""

js_code_4 = """
() => {
    let moment_text = document.getElementById('result_4').textContent;
    var replaced_text = moment_text.replace(/moment..../, '').replace(/\ Score.*/, '');
    let start_end = JSON.parse(replaced_text);

    document.getElementsByTagName("video")[0].currentTime = start_end[0];
    document.getElementsByTagName("video")[0].play();
}
"""

js_code_5 = """
() => {
    let moment_text = document.getElementById('result_5').textContent;
    var replaced_text = moment_text.replace(/moment..../, '').replace(/\ Score.*/, '');
    let start_end = JSON.parse(replaced_text);

    document.getElementsByTagName("video")[0].currentTime = start_end[0];
    document.getElementsByTagName("video")[0].play();
}
"""

def predict(video, textbox, button_1, button_2, button_3, button_4, button_5, line):
    model.encode_video(video)
    prediction = model.retrieve(textbox)
    mr_results = prediction['pred_relevant_windows']
    hl_results = prediction['pred_saliency_scores']

    buttons = []
    for i, pred in enumerate(mr_results[:5]):
        buttons.append(gr.Button(value='moment {}: [{}, {}] Score: {}'.format(i+1, pred[0], pred[1], pred[2]), visible=True))
    
    # Visualize the HD score
    seconds = [model.clip_len * i for i in range(len(hl_results))]
    hl_data = pd.DataFrame({ 'second': seconds, 'saliency_score': hl_results })
    min_val, max_val = min(hl_results), max(hl_results)+1
    min_x, max_x = min(seconds), max(seconds)
    line = gr.LinePlot(value=hl_data, x='second', y='saliency_score', visible=True, y_lim=[min_val, max_val], x_lim=[min_x, max_x])

    return buttons + [line]

def main():
    title = """# Moment Retrieval & Highlight Detection Demo: CG-DETR trained on QVHighlights"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(title)

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    video_input = gr.Video(elem_id='video', height=600)
                    query_input = gr.Textbox(label='query')
                    button = gr.Button("Retrieve moment & highlight detection", variant="primary")
            
            with gr.Column():
                with gr.Group():
                    gr.Markdown("## Retrieved moments")
                    button_1 = gr.Button(value='moment 1', visible=False, elem_id='result_1')
                    button_2 = gr.Button(value='moment 2', visible=False, elem_id='result_2')
                    button_3 = gr.Button(value='moment 3', visible=False, elem_id='result_3')
                    button_4 = gr.Button(value='moment 4', visible=False, elem_id='result_4')
                    button_5 = gr.Button(value='moment 5', visible=False, elem_id='result_5')

                    button_1.click(None, None, None, js=js_code_1)
                    button_2.click(None, None, None, js=js_code_2)
                    button_3.click(None, None, None, js=js_code_3)
                    button_4.click(None, None, None, js=js_code_4)
                    button_5.click(None, None, None, js=js_code_5)

                # dummy
                with gr.Group():
                    gr.Markdown("## Saliency score")
                    data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]})
                    line = gr.LinePlot(value=pd.DataFrame({'x': [], 'y': []}), x='x', y='y', visible=False)
                
                button.click(predict, inputs=[video_input, query_input, button_1, button_2, button_3, button_4, button_5, line], 
                            outputs=[button_1, button_2, button_3, button_4, button_5, line])

    demo.launch()

if __name__ == "__main__":
    main()