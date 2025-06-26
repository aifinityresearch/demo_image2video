import os
import uuid
import subprocess
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
CURRENT_DIR = "/home/ubuntu/VACE/demo_image2video/server"
VACE_ROOT = "/home/ubuntu/VACE"
MODEL_NAME = "vace1.3b"  # Matches folder name under results
CKPT_DIR = os.path.join(VACE_ROOT, "models/Wan2.1-VACE-1.3B")
IMAGE_DIR = os.path.join(CURRENT_DIR, "assets/images")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results", MODEL_NAME)

@app.post("/generate_video/")
async def generate_video(
    prompt: str = Form(...),
    father_img: UploadFile = File(...),
    baby_img: UploadFile = File(...),
    size: str = Form("480p"),
    frame_num: int = Form(81),
):
    session_id = uuid.uuid4().hex
    session_dir = os.path.join(IMAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Save images
    father_path = os.path.join(session_dir, "father.jpg")
    baby_path = os.path.join(session_dir, "baby.jpg")
    with open(father_path, "wb") as f:
        f.write(await father_img.read())
    with open(baby_path, "wb") as f:
        f.write(await baby_img.read())

    # Command to run inference from /home/ubuntu/VACE
    command = [
        "python3", "vace/vace_wan_inference.py",
        "--model_name", MODEL_NAME,
        "--ckpt_dir", CKPT_DIR,
        "--prompt", prompt,
        "--src_ref_images", f"{father_path},{baby_path}",
        "--size", size,
        "--frame_num", str(frame_num),
    ]

    # Run command
    process = subprocess.run(command, cwd=VACE_ROOT, capture_output=True, text=True)
    if process.returncode != 0:
        return {"error": "Video generation failed", "details": process.stderr}

    # Find latest result folder
    subdirs = [os.path.join(RESULTS_DIR, d) for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    if not subdirs:
        return {"error": "No output folder found in results."}
    latest_dir = max(subdirs, key=os.path.getmtime)

    video_path = os.path.join(latest_dir, "out_video.mp4")
    if not os.path.exists(video_path):
        return {"error": "out_video.mp4 not found in latest results directory."}

    return FileResponse(video_path, media_type="video/mp4", filename="generated_video.mp4")
