from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import subprocess
import os
import uuid
import uvicorn
import tempfile
import asyncio

app = FastAPI()

# Paths
SCRIPT_PATH = "/home/ubuntu/VACE/vace/vace_wan_inference.py"
MODEL_DIR = "/home/ubuntu/VACE/models/Wan2.1-VACE-1.3B"
OUTPUT_DIR = os.getcwd()

@app.post("/generate_i2v")
async def generate_video(prompt: str = Form(...),
                         image1: UploadFile = File(...),
                         image2: UploadFile = File(None)):

    try:
        print(f"[INFO] Received prompt: {prompt}")

        # Create temporary files for uploaded images
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img1:
            temp_img1.write(await image1.read())
            img1_path = temp_img1.name

        image_paths = [img1_path]

        if image2:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img2:
                temp_img2.write(await image2.read())
                img2_path = temp_img2.name
                image_paths.append(img2_path)

        src_refs = ",".join(image_paths)
        existing_files = set(os.listdir(OUTPUT_DIR))

        cmd = [
            "python3", SCRIPT_PATH,
            "--model_name", "vace-1.3B",
            "--ckpt_dir", MODEL_DIR,
            "--prompt", prompt,
            "--src_ref_images", src_refs,
            "--size", "480p",
            "--frame_num", "81"
        ]

        print(f"[INFO] Running command asynchronously: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            print(f"[ERROR] Generation failed:\n{stderr.decode()}")
            return JSONResponse(status_code=500, content={"error": stderr.decode()})

        new_files = set(os.listdir(OUTPUT_DIR)) - existing_files
        mp4_files = [f for f in new_files if f.endswith(".mp4")]

        if not mp4_files:
            return JSONResponse(status_code=500, content={"error": "No .mp4 file was created."})

        latest_file = max(
            [os.path.join(OUTPUT_DIR, f) for f in mp4_files],
            key=os.path.getctime
        )
        print(f"[SUCCESS] Returning file: {latest_file}")

        return FileResponse(latest_file, media_type="video/mp4", filename=os.path.basename(latest_file))

    except Exception as e:
        print(f"[EXCEPTION] {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860)
