import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import gc
import pandas as pd
from mmengine.config import Config
from mmseg.apis import init_model
from mmengine.dataset import default_collate
from mmengine.registry import DATASETS
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

# ========== CLASS LABELS ==========
CLASS_LABELS = [
    "background",
    "apple black rot", "apple mosaic virus", "apple rust", "apple scab",
    "banana anthracnose", "banana black leaf streak", "banana bunchy top",
    "banana cigar end rot", "banana cordana leaf spot", "banana panama disease",
    "basil downy mildew", "bean halo blight", "bean mosaic virus", "bean rust",
    "bell pepper bacterial spot", "bell pepper blossom end rot", "bell pepper frogeye leaf spot", "bell pepper powdery mildew",
    "blueberry anthracnose", "blueberry botrytis blight", "blueberry mummy berry", "blueberry rust", "blueberry scorch",
    "broccoli alternaria leaf spot", "broccoli downy mildew", "broccoli ring spot",
    "cabbage alternaria leaf spot", "cabbage black rot", "cabbage downy mildew",
    "carrot alternaria leaf blight", "carrot cavity spot", "carrot cercospora leaf blight",
    "cauliflower alternaria leaf spot", "cauliflower bacterial soft rot",
    "celery anthracnose", "celery early blight",
    "cherry leaf spot", "cherry powdery mildew",
    "citrus canker", "citrus greening disease",
    "coffee berry blotch", "coffee black rot", "coffee brown eye spot", "coffee leaf rust",
    "corn gray leaf spot", "corn northern leaf blight", "corn rust", "corn smut",
    "cucumber angular leaf spot", "cucumber bacterial wilt", "cucumber powdery mildew",
    "eggplant cercospora leaf spot", "eggplant phomopsis fruit rot", "eggplant phytophthora blight",
    "garlic leaf blight", "garlic rust", "ginger leaf spot", "ginger sheath blight",
    "grape black rot", "grape downy mildew", "grape leaf spot", "grapevine leafroll disease",
    "lettuce downy mildew", "lettuce mosaic virus", "maple tar spot",
    "peach anthracnose", "peach brown rot", "peach leaf curl", "peach rust", "peach scab",
    "plum bacterial spot", "plum brown rot", "plum pocket disease", "plum pox virus", "plum rust",
    "potato early blight", "potato late blight",
    "raspberry fire blight", "raspberry gray mold", "raspberry leaf spot", "raspberry yellow rust",
    "rice blast", "rice sheath blight",
    "soybean bacterial blight", "soybean brown spot", "soybean downy mildew", "soybean frog eye leaf spot", "soybean mosaic", "soybean rust",
    "squash powdery mildew",
    "strawberry anthracnose", "strawberry leaf scorch",
    "tobacco blue mold", "tobacco brown spot", "tobacco frogeye leaf spot", "tobacco mosaic virus",
    "tomato bacterial leaf spot", "tomato early blight", "tomato late blight", "tomato leaf mold",
    "tomato mosaic virus", "tomato septoria leaf spot", "tomato yellow leaf curl virus",
    "wheat bacterial leaf streak (black chaff)", "wheat head scab", "wheat leaf rust",
    "wheat loose smut", "wheat powdery mildew", "wheat septoria blotch", "wheat stem rust", "wheat stripe rust",
    "zucchini bacterial wilt", "zucchini downy mildew", "zucchini powdery mildew", "zucchini yellow mosaic virus"
]
num_classes = len(CLASS_LABELS)

# ========== Step 1: Generate annotation_test.txt ==========
json_path = 'data/plantseg115/annotation_test.json'
txt_path = 'annotation_test.txt'

with open(json_path, 'r') as f:
    ann = json.load(f)

lines = []
for img in ann['images']:
    stem = img['file_name'].rsplit('.', 1)[0]
    lines.append(f'test/{stem}')

with open(txt_path, 'w') as f:
    f.write('\n'.join(lines))

print(f"[✓] Saved annotation_test.txt with {len(lines)} entries.")

# ========== Step 2: Patch config ==========
config_path = 'configs/segnext/segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512.py'
checkpoint_path = 'work_dirs/segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512/iter_8000.pth'

cfg = Config.fromfile(config_path)
cfg.test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=cfg.test_dataloader.dataset.type,
        data_root='',
        ann_file='annotation_test.txt',
        data_prefix=dict(
            img_path='data/plantseg115/images',
            seg_map_path='data/plantseg115/annotations'
        ),
        pipeline=cfg.test_dataloader.dataset.pipeline
    )
)
cfg.test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mAcc', 'aAcc'])

# ========== Step 3: Init CUDA model ==========
print("🚀 Loading model on CUDA...")
model = init_model(cfg, checkpoint_path, device='cuda')
model.eval()

# ========== Step 4: Inference with OOM fallback ==========
test_dataset = DATASETS.build(cfg.test_dataloader.dataset)
all_preds, all_gts = [], []
fallback_cpu = []

for i in range(len(test_dataset)):
    data = test_dataset[i]
    batch = default_collate([data])
    try:
        with torch.no_grad():
            result = model.test_step(batch)[0]
    except torch.cuda.OutOfMemoryError:
        print(f"❌ OOM at index {i} → fallback to CPU")
        torch.cuda.empty_cache()
        gc.collect()
        model_cpu = init_model(cfg, checkpoint_path, device='cpu')
        model_cpu.eval()
        result = model_cpu.test_step(batch)[0]
        fallback_cpu.append(i)
        del model_cpu

    pred = result.pred_sem_seg.data.squeeze().cpu().numpy()
    gt = data['data_samples'].gt_sem_seg.data.squeeze().cpu().numpy()
    all_preds.append(pred.flatten())
    all_gts.append(gt.flatten())

    del result, batch
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n✅ Inference complete. Fallback to CPU on {len(fallback_cpu)} samples.")

# ========== Step 5: Metrics ==========
all_preds = np.concatenate(all_preds)
all_gts = np.concatenate(all_gts)

cm = confusion_matrix(all_gts, all_preds, labels=list(range(num_classes)))
acc = np.mean(all_preds == all_gts)
print(f"\n🎯 Overall Accuracy: {acc * 100:.2f}%")

# ========= Step 6: Per-class metrics ==========
precision = precision_score(all_gts, all_preds, labels=list(range(num_classes)), average=None, zero_division=0)
recall = recall_score(all_gts, all_preds, labels=list(range(num_classes)), average=None, zero_division=0)

# ========== Step 7: CSV & Display ==========
df = pd.DataFrame({
    'Class': CLASS_LABELS,
    'Precision': precision,
    'Recall': recall,
    'Support': np.bincount(all_gts, minlength=num_classes)
})
df.to_csv("confusion_metrics.csv", index=False)
print("✅ Saved per-class metrics to confusion_metrics.csv")

# ========== Step 8: Confusion Matrix Plot ==========
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=90)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("✅ Confusion matrix saved to confusion_matrix.png")
