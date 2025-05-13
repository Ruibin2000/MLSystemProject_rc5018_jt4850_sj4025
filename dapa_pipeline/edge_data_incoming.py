import requests
import os
import time
from glob import glob

FAST_URL = "http://192.5.86.161:8500/predict"  # place holder for your fast api
image_dir = "/mnt/object-uc/images/train/"  # our data

image_paths = glob(os.path.join(image_dir, "*.jpg"))#get a pic

for image_path in image_paths:
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(FAST_URL, files=files)
    
    if response.status_code == 200:
        print(f"[OK] {os.path.basename(image_path)}")
    else:
        print(f"[FAIL] {os.path.basename(image_path)} - Status {response.status_code}")
    
    time.sleep(30)  # consider edge device is weak so we sleep for 30 s to get next picture.

