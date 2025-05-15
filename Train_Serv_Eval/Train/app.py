import os
import time
from io import BytesIO
from typing import Optional, Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse
from mmseg.apis import init_model, inference_model
from prometheus_fastapi_instrumentator import Instrumentator
matplotlib.use("Agg")

# --------------------------------------------------
# 路径配置（本地或 Docker 环境）
# --------------------------------------------------
MODEL_DIR = "."
DEFAULT_ONNX = "model"
DEFAULT_PTH = "model.pth"
DEFAULT_CFG = "PlantSeg/configs/segnext/segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512.py"
MODEL_EXT = ".onnx"
PORT = 8000

app = FastAPI(title="PlantSeg Serving", version="0.1.5")

_sessions: dict[str, ort.InferenceSession] = {}
_pth_model = None

# --------------------------------------------------
# ONNX 推理专用预处理（输出与原图一致）
# --------------------------------------------------
def onnx_preprocess_with_size(img_bytes: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    rgb_img = cv2.resize(rgb_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    norm_img = (rgb_img - mean) / std
    norm_img = np.transpose(norm_img, (2, 0, 1))[None, ...]  # NCHW

    return norm_img.astype(np.float32), (w, h)

# --------------------------------------------------
# 工具函数
# --------------------------------------------------
def get_session(name: str) -> ort.InferenceSession:
    if name in _sessions:
        return _sessions[name]

    path = os.path.join(MODEL_DIR, f"{name}{MODEL_EXT}")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"找不到 ONNX 模型: {path}")

    try:
        sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
    except Exception:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    _sessions[name] = sess
    return sess

def colormap(mask: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("tab20", 116)
    rgb = cmap(mask / 115.0)[..., :3] * 255
    return rgb.astype(np.uint8)

def _decode_mmseg(result: Any) -> np.ndarray:
    if isinstance(result, np.ndarray):
        return result

    try:
        from mmengine.structures import BaseDataElement
    except Exception:
        BaseDataElement = object

    if isinstance(result, list):
        result = result[0]

    if hasattr(result, "pred_sem_seg"):
        tensor = result.pred_sem_seg.data
        return tensor.squeeze().cpu().numpy()

    raise TypeError("Unsupported mmseg result type: %s" % type(result))

# --------------------------------------------------
# FastAPI 路由
# --------------------------------------------------

@app.post("/predict/", summary="ONNX → JSON 结果")
async def predict(
    file: UploadFile = File(..., description="上传图片"),
    model_name: str = Query(DEFAULT_ONNX, description="不含 .onnx 的文件名"),
):
    sess = get_session(model_name)
    img, _ = onnx_preprocess_with_size(await file.read())
    tic = time.time()
    out = sess.run(None, {sess.get_inputs()[0].name: img})[0]
    latency = (time.time() - tic) * 1000
    preds = np.argmax(out[0], axis=0)
    unique, counts = np.unique(preds, return_counts=True)
    class_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
    non_bg = {k: v for k, v in class_distribution.items() if k != 0}
    top_class = max(non_bg.items(), key=lambda x: x[1])[0] if non_bg else None
    return {
        "model": model_name,
        "shape": preds.shape,
        "classes": int(out.shape[1]),
        "latency_ms": round(latency, 2),
        "class_distribution": class_distribution,
        "top_non_background": top_class,
    }

@app.post("/predict_mask/", summary="ONNX → 彩色掩码图")
async def predict_mask(
    file: UploadFile = File(...),
    model_name: str = Query(DEFAULT_ONNX),
):
    sess = get_session(model_name)
    img, (orig_w, orig_h) = onnx_preprocess_with_size(await file.read())
    out = sess.run(None, {sess.get_inputs()[0].name: img})[0]
    mask = np.argmax(out[0], axis=0)
    mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), cv2.INTER_NEAREST)
    color = colormap(mask)
    _, buf = cv2.imencode(".png", color)
    return StreamingResponse(BytesIO(buf.tobytes()), media_type="image/png")

@app.get("/list_models", summary="列出可用 ONNX")
def list_models() -> dict:
    models = [f[:-len(MODEL_EXT)] for f in os.listdir(MODEL_DIR) if f.endswith(MODEL_EXT)]
    return {"onnx_models": models}

# -------- PTH 支持，懒加载 ---------

def _load_pth():
    global _pth_model
    if _pth_model is not None:
        return _pth_model

    if not (os.path.isfile(DEFAULT_PTH) and os.path.isfile(DEFAULT_CFG)):
        raise HTTPException(status_code=404, detail="PTH 或 config 文件未找到，请检查挂载路径")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        _pth_model = init_model(DEFAULT_CFG, DEFAULT_PTH, device=device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载 PTH 失败: {e}") from e
    return _pth_model

@app.post("/predict_pth/", summary="PTH → JSON 结果")
async def predict_pth(file: UploadFile = File(...)):
    img_bytes = await file.read()
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    model = _load_pth()
    tic = time.time()
    raw = inference_model(model, img)
    mask = _decode_mmseg(raw)
    latency = (time.time() - tic) * 1000
    unique, counts = np.unique(mask, return_counts=True)
    class_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
    non_bg = {k: v for k, v in class_distribution.items() if k != 0}
    top_class = max(non_bg.items(), key=lambda x: x[1])[0] if non_bg else None
    return {
        "model": os.path.basename(DEFAULT_PTH),
        "shape": mask.shape,
        "classes": int(mask.max()) + 1,
        "latency_ms": round(latency, 2),
        "class_distribution": class_distribution,
        "top_non_background": top_class,
    }

@app.post("/predict_pth_mask/", summary="PTH → 彩色掩码图")
async def predict_pth_mask(file: UploadFile = File(...)):
    img_bytes = await file.read()
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    model = _load_pth()
    raw = inference_model(model, img)
    mask = _decode_mmseg(raw).astype(np.uint8)
    color = colormap(mask)
    _, buf = cv2.imencode(".png", color)
    return StreamingResponse(BytesIO(buf.tobytes()), media_type="image/png")

Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)
