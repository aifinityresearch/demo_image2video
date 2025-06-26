import gradio as gr
import requests

API_URL = "http://13.232.209.100:8000/generate_video/"  # Replace with your real server IP

def generate_video(prompt, image_path):
    if not prompt or not image_path:
        return None, "❗ Please provide both a prompt and a reference image."

    try:
        files = {
            "ref_img": open(image_path, "rb"),
        }
        data = {
            "prompt": prompt,
            "size": "480p",
            "frame_num": "81",
        }

        response = requests.post(API_URL, data=data, files=files)

        if response.status_code == 200:
            with open("generated_video.mp4", "wb") as f:
                f.write(response.content)
            return "generated_video.mp4", "✅ Video generated!"
        else:
            return None, f"❌ Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"❌ Request failed: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 AIfinity Research : Image-to-Video Generator - WAN VACE 1.3B")
    prompt = gr.Textbox(label="Prompt")
    image = gr.Image(label="Reference Image", type="filepath")
    generate_btn = gr.Button("Generate Video")

    video_output = gr.Video()
    status_output = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate_video,
        inputs=[prompt, image],
        outputs=[video_output, status_output],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
