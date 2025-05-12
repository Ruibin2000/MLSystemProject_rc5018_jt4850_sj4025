import requests
import os
import time
from glob import glob

FLASK_URL = "http://your_flask_server:5000/predict"  # place holder for api address which I dont have now
image_dir = "/mnt/object/images/train/"  # our data

image_paths = glob(os.path.join(image_dir, "*.jpg"))#get a pic

for image_path in image_paths:
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(FLASK_URL, files=files)
    
    if response.status_code == 200:
        print(f"[OK] {os.path.basename(image_path)}")
    else:
        print(f"[FAIL] {os.path.basename(image_path)} - Status {response.status_code}")
    
    time.sleep(30)  # consider edge device is weak so we sleep for 30 s to get next picture.

