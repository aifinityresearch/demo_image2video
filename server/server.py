import os
import sys
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

# Paths relative to where main.py is executed
CURRENT_DIR = os.getcwd()
MODEL_NAME = "vace-1.3B"
VACE_WORK_DIR = "/home/ubuntu/VACE"
IMAGE_DIR = os.path.join(CURRENT_DIR, "assets/images")
RESULTS_DIR = os.path.join(VACE_WORK_DIR, "results", MODEL_NAME)
CKPT_DIR = "/home/ubuntu/VACE/models/Wan2.1-VACE-1.3B"
VACE_SCRIPT_PATH = "/home/ubuntu/VACE/vace/vace_wan_inference.py"

@app.post("/generate_video/")
async def generate_video(
    prompt: str = Form(...),
    ref_img: UploadFile = File(...),
    size: str = Form("480p"),
    frame_num: int = Form(81),
):
    print("📥 Received request to generate video.", flush=True)
    print(f"Prompt: {prompt}", flush=True)
    print(f"Current Working Directory: {CURRENT_DIR}", flush=True)

    session_id = uuid.uuid4().hex
    session_dir = os.path.join(IMAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    print(f"📁 Created session directory: {session_dir}", flush=True)

    ref_path = os.path.join(session_dir, f"img_{uuid.uuid4().hex[:6]}.jpg")
   

    with open(ref_path, "wb") as f1:
        data1 = await ref_img.read()
        f1.write(data1)
    
    print(f"🖼️ Saved ref image at: {ref_path}", flush=True)
    
    command = [
        "python3", VACE_SCRIPT_PATH,
        "--model_name", MODEL_NAME,
        "--ckpt_dir", CKPT_DIR,
        "--prompt", prompt,
        "--src_ref_images", ref_path,
        "--size", size,
        "--frame_num", str(frame_num),
    ]
    print(f"🚀 Running command:\n{' '.join(command)}", flush=True)

    #process = subprocess.run(command, cwd="/home/ubuntu/VACE", capture_output=True, text=True)
    process = subprocess.run(command, cwd="/home/ubuntu/VACE", stdout=sys.stdout, stderr=sys.stderr, text=True)
    print("✅ Model command executed.", flush=True)
    print("STDOUT:\n", process.stdout, flush=True)
    print("STDERR:\n", process.stderr, flush=True)

    if process.returncode != 0:
        print("❌ Video generation failed.", flush=True)
        return {"error": "Video generation failed", "details": process.stderr}

    if not os.path.exists(RESULTS_DIR):
        print(f"❌ Results directory not found: {RESULTS_DIR}", flush=True)
        return {"error": f"Results directory not found: {RESULTS_DIR}"}

    subdirs = [os.path.join(RESULTS_DIR, d) for d in os.listdir(RESULTS_DIR)
               if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    if not subdirs:
        print("❌ No output folders found in results directory.", flush=True)
        return {"error": "No output folders in results directory."}

    latest_dir = max(subdirs, key=os.path.getmtime)
    print(f"📂 Latest result directory: {latest_dir}", flush=True)

    video_path = os.path.join(latest_dir, "out_video.mp4")
    if not os.path.exists(video_path):
        print(f"❌ Video file not found at: {video_path}", flush=True)
        return {"error": f"Video file not found at: {video_path}"}

    print(f"✅ Video successfully generated at: {video_path}", flush=True)
    return FileResponse(video_path, media_type="video/mp4", filename="generated_video.mp4")
