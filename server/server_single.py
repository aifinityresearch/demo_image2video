from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import os
import subprocess
import uuid
import glob

app = FastAPI()

IMAGE_DIR = "/home/ubuntu/VACE/demo_image2video/server/uploaded_images"
RESULTS_DIR = "/home/ubuntu/VACE/demo_image2video/server/results/vace-1.3B"
VACE_SCRIPT = "/home/ubuntu/VACE/vace/vace_wan_inference.py"
MODEL_NAME = "vace-1.3B"
CKPT_DIR = "/home/ubuntu/VACE/models/Wan2.1-VACE-1.3B"
DEFAULT_SIZE = "480p"
DEFAULT_FRAMES = "81"

os.makedirs(IMAGE_DIR, exist_ok=True)

@app.post("/generate-video/")
async def generate_video(prompt: str = Form(...), image: UploadFile = File(...)):
    print("Received a video generation request.")

    # Save uploaded image
    image_id = str(uuid.uuid4())
    image_filename = f"{image_id}_{image.filename}"
    image_path = os.path.join(IMAGE_DIR, image_filename)

    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)

    print(f"Image saved: {image_path}")
    print(f"Prompt: {prompt}")

    # Run generation script
    cmd = [
        "python3", VACE_SCRIPT,
        "--model_name", MODEL_NAME,
        "--ckpt_dir", CKPT_DIR,
        "--prompt", prompt,
        "--src_ref_images", image_path,
        "--size", DEFAULT_SIZE,
        "--frame_num", DEFAULT_FRAMES
    ]
    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Generation script failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    # Search for the latest out_video.mp4 in timestamped result folders
    video_candidates = glob.glob(os.path.join(RESULTS_DIR, "*", "out_video.mp4"))
    if not video_candidates:
        print("No output video file found.")
        return JSONResponse(status_code=500, content={"error": "Video generation failed."})

    latest_file = max(video_candidates, key=os.path.getctime)
    print(f"Returning generated video: {latest_file}")

    return FileResponse(latest_file, media_type="video/mp4", filename=os.path.basename(latest_file))
