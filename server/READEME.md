# To run the server follow steps below

pip install -r requirements.txt

Login to AWS console --> EC2 Connect to your g5.2xlarge instance

cd /home/ubuntu/VACE

git clone https://github.com/aifinityresearch/demo_image2video.git

cd /demo_image2video/server
# Run the server in backgrund 
nohup uvicorn server:app --host 0.0.0.0 --port 7860 > log.log 2>&1 &
# To check status and debug messages
tail -f log.log 
# To kill the running instance 
ps -aux | grep python # note down the PID
kill PID

# To download .mp4 output manually from EC2 to laptop . Run from Windows Powershell 

scp -i WAN_Server.pem ubuntu@13.232.209.100:/home/ubuntu/VACE/results/vace-1.3B/2025-06-26-06-10-57/out_video.mp4 .

# To check if model is running good or not 
cd /VACE
python3 vace/vace_wan_inference.py \
  --model_name vace-1.3B \
  --ckpt_dir ./models/Wan2.1-VACE-1.3B \
  --prompt "A realistic video of a father and daughter celebrating her birthday in a boat, where the father and daughter closely resemble the faces from the provided reference images." \
  --src_ref_images assets/images/father_closeup.jpg,assets/images/baby_closeup.jpg \
  --size 480p \
  --frame_num 81




