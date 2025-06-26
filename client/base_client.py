import requests

API_URL = "http://13.232.209.100:8000/generate_video/"  # Replace with your real server IP

def generate_video(prompt, image_path):
    if not prompt or not image_path:
        print("❗ Please provide both a prompt and a reference image.")
        return

    try:
        with open(image_path, "rb") as img_file:
            files = {
                "ref_img": img_file,
            }
            data = {
                "prompt": prompt,
                "size": "480p",
                "frame_num": "81",
            }

            print("📡 Sending request to server...")
            response = requests.post(API_URL, data=data, files=files)

            if response.status_code == 200:
                with open("generated_video.mp4", "wb") as f:
                    f.write(response.content)
                print("✅ Video successfully generated and saved as 'generated_video.mp4'")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")

if __name__ == "__main__":
    # Modify the prompt and image path as needed
    prompt = "An young indian girl flying on a UFO in a futuristic city"
    image_path = "input/sanjhana1.jpg"

    generate_video(prompt, image_path)
