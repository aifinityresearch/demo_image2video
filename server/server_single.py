from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import subprocess
import uuid

app = FastAPI()

OUTPUT_DIR = "./generated_videos"
IMAGE_DIR = "./uploaded_images"
VACE_SCRIPT = "vace/vace_wan_inference.py"
MODEL_NAME = "vace-1.3B"
CKPT_DIR = "./models/Wan2.1-VACE-1.3B"
DEFAULT_SIZE = "480p"
DEFAULT_FRAMES = "81"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

@app.post("/generate-video/")
async def generate_video(prompt: str = Form(...), image: UploadFile = File(...)):
    print("Received a new video generation request", flush=True)

    image_id = str(uuid.uuid4())
    image_path = os.path.join(IMAGE_DIR, f"{image_id}_{image.filename}")
    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)
    print(f"Saved uploaded image to: {image_path}", flush=True)
    print(f"Prompt received: {prompt}", flush=True)

    cmd = [
        "python3", VACE_SCRIPT,
        "--model_name", MODEL_NAME,
        "--ckpt_dir", CKPT_DIR,
        "--prompt", prompt,
        "--src_ref_images", image_path,
        "--size", DEFAULT_SIZE,
        "--frame_num", DEFAULT_FRAMES
    ]
    print(f"Executing command: {' '.join(cmd)}", flush=True)

    try:
        subprocess.run(cmd, check=True)
        print("Video generation script executed successfully.", flush=True)
        return JSONResponse(content={"message": "Video generation completed successfully."})
    except subprocess.CalledProcessError as e:
        print(f"Video generation failed: {e}", flush=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Video generation failed: {e}"}
        )
