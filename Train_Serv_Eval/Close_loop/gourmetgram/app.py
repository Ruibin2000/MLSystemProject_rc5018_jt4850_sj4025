import os
import uuid
import boto3
import requests
import numpy as np
from datetime import datetime
from flask import Flask, request, send_file
from mimetypes import guess_type
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)

executor = ThreadPoolExecutor(max_workers=2)

# --------------------------
# Class ID → Name mapping
# --------------------------
CLASS_LABELS = {
    0: "background",
    1: "apple black rot",
    2: "apple mosaic virus",
    3: "apple rust",
    4: "apple scab",
    5: "banana anthracnose",
    6: "banana black leaf streak",
    7: "banana bunchy top",
    8: "banana cigar end rot",
    9: "banana cordana leaf spot",
    10: "banana panama disease",
    11: "basil downy mildew",
    12: "bean halo blight",
    13: "bean mosaic virus",
    14: "bean rust",
    15: "bell pepper bacterial spot",
    16: "bell pepper blossom end rot",
    17: "bell pepper frogeye leaf spot",
    18: "bell pepper powdery mildew",
    19: "blueberry anthracnose",
    20: "blueberry botrytis blight",
    21: "blueberry mummy berry",
    22: "blueberry rust",
    23: "blueberry scorch",
    24: "broccoli alternaria leaf spot",
    25: "broccoli downy mildew",
    26: "broccoli ring spot",
    27: "cabbage alternaria leaf spot",
    28: "cabbage black rot",
    29: "cabbage downy mildew",
    30: "carrot alternaria leaf blight",
    31: "carrot cavity spot",
    32: "carrot cercospora leaf blight",
    33: "cauliflower alternaria leaf spot",
    34: "cauliflower bacterial soft rot",
    35: "celery anthracnose",
    36: "celery early blight",
    37: "cherry leaf spot",
    38: "cherry powdery mildew",
    39: "citrus canker",
    40: "citrus greening disease",
    41: "coffee berry blotch",
    42: "coffee black rot",
    43: "coffee brown eye spot",
    44: "coffee leaf rust",
    45: "corn gray leaf spot",
    46: "corn northern leaf blight",
    47: "corn rust",
    48: "corn smut",
    49: "cucumber angular leaf spot",
    50: "cucumber bacterial wilt",
    51: "cucumber powdery mildew",
    52: "eggplant cercospora leaf spot",
    53: "eggplant phomopsis fruit rot",
    54: "eggplant phytophthora blight",
    55: "garlic leaf blight",
    56: "garlic rust",
    57: "ginger leaf spot",
    58: "ginger sheath blight",
    59: "grape black rot",
    60: "grape downy mildew",
    61: "grape leaf spot",
    62: "grapevine leafroll disease",
    63: "lettuce downy mildew",
    64: "lettuce mosaic virus",
    65: "maple tar spot",
    66: "peach anthracnose",
    67: "peach brown rot",
    68: "peach leaf curl",
    69: "peach rust",
    70: "peach scab",
    71: "plum bacterial spot",
    72: "plum brown rot",
    73: "plum pocket disease",
    74: "plum pox virus",
    75: "plum rust",
    76: "potato early blight",
    77: "potato late blight",
    78: "raspberry fire blight",
    79: "raspberry gray mold",
    80: "raspberry leaf spot",
    81: "raspberry yellow rust",
    82: "rice blast",
    83: "rice sheath blight",
    84: "soybean bacterial blight",
    85: "soybean brown spot",
    86: "soybean downy mildew",
    87: "soybean frog eye leaf spot",
    88: "soybean mosaic",
    89: "soybean rust",
    90: "squash powdery mildew",
    91: "strawberry anthracnose",
    92: "strawberry leaf scorch",
    93: "tobacco blue mold",
    94: "tobacco brown spot",
    95: "tobacco frogeye leaf spot",
    96: "tobacco mosaic virus",
    97: "tomato bacterial leaf spot",
    98: "tomato early blight",
    99: "tomato late blight",
    100: "tomato leaf mold",
    101: "tomato mosaic virus",
    102: "tomato septoria leaf spot",
    103: "tomato yellow leaf curl virus",
    104: "wheat bacterial leaf streak (black chaff)",
    105: "wheat head scab",
    106: "wheat leaf rust",
    107: "wheat loose smut",
    108: "wheat powdery mildew",
    109: "wheat septoria blotch",
    110: "wheat stem rust",
    111: "wheat stripe rust",
    112: "zucchini bacterial wilt",
    113: "zucchini downy mildew",
    114: "zucchini powdery mildew",
    115: "zucchini yellow mosaic virus"
}

# --------------------------
# MinIO setup
# --------------------------
s3 = boto3.client(
    's3',
    endpoint_url=os.environ['MINIO_URL'],
    aws_access_key_id=os.environ['MINIO_USER'],
    aws_secret_access_key=os.environ['MINIO_PASSWORD'],
    region_name='us-east-1'
)
bucket_name = "plantseg-upload"

# --------------------------
# Upload to MinIO
# --------------------------
def upload_production_bucket(img_path, predicted_class, confidence, prediction_id, mask_path):
    class_index = list(CLASS_LABELS.values()).index(predicted_class)
    class_dir = f"class_{class_index:02d}"
    ext = os.path.splitext(img_path)[-1]
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    s3_key_img = f"{class_dir}/{prediction_id}{ext}"
    s3_key_mask = f"{class_dir}/{prediction_id}_mask.png"

    # Upload image
    with open(img_path, 'rb') as f:
        s3.upload_fileobj(f, bucket_name, s3_key_img, ExtraArgs={
            'ContentType': guess_type(img_path)[0] or 'application/octet-stream'
        })

    # Upload segmentation mask
    with open(mask_path, 'rb') as f:
        s3.upload_fileobj(f, bucket_name, s3_key_mask, ExtraArgs={
            'ContentType': 'image/png'
        })

    # Add tag to image (not mask)
    s3.put_object_tagging(
        Bucket=bucket_name,
        Key=s3_key_img,
        Tagging={
            'TagSet': [
                {'Key': 'predicted_class', 'Value': predicted_class},
                {'Key': 'confidence', 'Value': f"{confidence:.3f}"},
                {'Key': 'timestamp', 'Value': timestamp}
            ]
        }
    )



# --------------------------
# Model Inference
# --------------------------
def model_predict(img_path):
    FASTAPI_SERVER_URL = os.getenv("FASTAPI_SERVER_URL", "http://localhost:8500")
    ENDPOINT = os.getenv("FASTAPI_PREDICT_ENDPOINT", "/predict_pth")
    MASK_ENDPOINT = os.getenv("FASTAPI_MASK_ENDPOINT", "/predict_pth_mask")

    with open(img_path, "rb") as f:
        image_data = f.read()
    files = {"file": (os.path.basename(img_path), image_data, "image/jpeg")}

    response = requests.post(f"{FASTAPI_SERVER_URL}{ENDPOINT}", files=files)
    response.raise_for_status()
    result = response.json()

    mask_response = requests.post(f"{FASTAPI_SERVER_URL}{MASK_ENDPOINT}", files=files)
    mask_path = os.path.join(app.instance_path, 'uploads', 'mask.png')
    with open(mask_path, "wb") as f:
        f.write(mask_response.content)

    dist_raw = result.get("class_distribution", {})
    dist = {CLASS_LABELS.get(int(k), f"Class {k}"): v for k, v in dist_raw.items()}
    dist_wo_bg = {k: v for k, v in dist.items() if k != "background"}

    top_id = result.get("top_non_background", -1)
    top_name = CLASS_LABELS.get(top_id, f"Class {top_id}")
    top_conf = max(dist_wo_bg.values()) if dist_wo_bg else 0.0

    return {
        "model": result.get("model", "unknown"),
        "shape": result.get("shape", []),
        "latency": result.get("latency_ms", "?"),
        "top_class": top_name,
        "top_conf": top_conf,
        "distribution": dist,
        "mask_path": mask_path
    }

# --------------------------
# Home
# --------------------------
@app.route("/")
def index():
    return '''
    <h2>Upload .jpg image for PlantSeg prediction</h2>
    <form method="post" action="/predict" enctype="multipart/form-data">
        <input type="file" name="file" accept=".jpg,.jpeg,.png" required><br><br>
        <input type="submit" value="Submit">
    </form>
    '''

# --------------------------
# Prediction
# --------------------------
@app.route("/predict", methods=["POST"])
def upload():
    f = request.files["file"]
    filename = secure_filename(f.filename)
    img_path = os.path.join(app.instance_path, 'uploads', filename)
    f.save(img_path)

    try:
        result = model_predict(img_path)
    except Exception as e:
        return f"<b>Error:</b> {e}"

    prediction_id = str(uuid.uuid4())
    executor.submit(
        upload_production_bucket,
        img_path,
        result['top_class'],
        result['top_conf'],
        prediction_id,
        result['mask_path']
    )

    dist_html = "<ul>" + "".join([f"<li>{k}: {v}</li>" for k, v in result["distribution"].items()]) + "</ul>"

    return f'''
    <h3>✅ Segmentation Result</h3>
    <b>Model:</b> {result["model"]}<br>
    <b>Shape:</b> {result["shape"]}<br>
    <b>Latency:</b> {result["latency"]:.2f} ms<br>
    <b>Top Class:</b> {result["top_class"]}<br><br>

    <b>Class Distribution:</b> {dist_html}

    <h4>Segmentation Mask:</h4>
    <img src="/mask" style="max-width:512px; border:1px solid #ccc;"><br><br>
    <a href="/">⬅ Upload Another Image</a>
    '''

@app.route("/mask")
def serve_mask():
    return send_file(os.path.join(app.instance_path, 'uploads', 'mask.png'), mimetype="image/png")

@app.route("/test")
def test():
    return "Flask test endpoint OK."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
