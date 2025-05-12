import requests
import base64
import os
from glob import glob

FASTAPI_URL = "http://your_fastapi_server:5000/predict"
image_dir = "/mnt/object/images/train/"
image_paths = glob(os.path.join(image_dir, "*.jpg"))

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# upload a lot of pic at one time 
batch_size = 100

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    encoded_images = [encode_image(p) for p in batch_paths]

    payload = {"images": encoded_images}
    response = requests.post(FASTAPI_URL, json=payload)

    if response.status_code == 200:
        print(f"[OK] {os.path.basename(image_path)}")
    else:
        print(f"[FAIL] {os.path.basename(image_path)} - Status {response.status_code}")
