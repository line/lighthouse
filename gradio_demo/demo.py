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

import gradio as gr
from lighthouse.models import CGDETRPredictor

model = CGDETRPredictor('results/clip_cg_detr/activitynet/best.ckpt', 
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

def moment_retrieval(video, textbox, button_1, button_2, button_3, button_4, button_5):
    model.encode_video(video)
    prediction = model.retrieve(textbox)
    prediction = prediction['pred_relevant_windows']

    buttons = []
    for i, pred in enumerate(prediction[:5]):
        buttons.append(gr.Button(value='moment {}: [{}, {}] Score: {}'.format(i+1, pred[0], pred[1], pred[2]), visible=True))
    return buttons

def main():
    title = """# Video Moment Retrieval Demo: CG-DETR trained on activitynet"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(title)

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    video_input = gr.Video(elem_id='video', height=600)
                    query_input = gr.Textbox(label='query')
                    button = gr.Button("Retrieve moment", variant="primary")
            
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
                
                button.click(moment_retrieval, inputs=[video_input, query_input, button_1, button_2, button_3, button_4, button_5], 
                            outputs=[button_1, button_2, button_3, button_4, button_5])

    demo.launch()

if __name__ == "__main__":
    main()