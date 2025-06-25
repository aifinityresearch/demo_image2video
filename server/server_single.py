from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import os
import subprocess
import uuid

app = FastAPI()

OUTPUT_DIR = "/home/ubuntu/VACE/demo_image2video/server/generated_videos"
IMAGE_DIR = "/home/ubuntu/VACE/demo_image2video/server/uploaded_images"
VACE_SCRIPT = "/home/ubuntu/VACE/vace/vace_wan_inference.py"
MODEL_NAME = "vace-1.3B"
CKPT_DIR = "/home/ubuntu/VACE/models/Wan2.1-VACE-1.3B"
DEFAULT_SIZE = "480p"
DEFAULT_FRAMES = "81"

os.makedirs(OUTPUT_DIR, exist_ok=True)
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

    # Track existing output files before execution
    existing_files = set(os.listdir(OUTPUT_DIR))

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
    print(f" Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Generation script failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    # Detect the new .mp4 file
    new_files = set(os.listdir(OUTPUT_DIR)) - existing_files
    mp4_files = [f for f in new_files if f.endswith(".mp4")]

    if not mp4_files:
        print("No output .mp4 found.")
        return JSONResponse(status_code=500, content={"error": "Video generation failed."})

    latest_file = max(
        [os.path.join(OUTPUT_DIR, f) for f in mp4_files],
        key=os.path.getctime
    )
    print(f"Returning generated video: {latest_file}")

    return FileResponse(latest_file, media_type="video/mp4", filename=os.path.basename(latest_file))
