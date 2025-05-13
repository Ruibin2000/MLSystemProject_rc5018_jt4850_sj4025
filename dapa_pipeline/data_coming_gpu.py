import requests
import base64
import os
from glob import glob

FAST_URL = "http://192.5.86.161:8500/predict"  # place holder for your fast api
image_dir = "/mnt/object-uc/images/train/"  # our data
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
    response = requests.post(FAST_URL, json=payload)

    if response.status_code == 200:
        for p in batch_paths:
            print(f"[OK] {os.path.basename(p)}")
    else:
        for p in batch_paths:
            print(f"[FAIL] {os.path.basename(p)} - Status {response.status_code}")
