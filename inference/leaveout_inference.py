import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from datasets.models.slowfast_model import SlowFastModel

# === CONFIG ===
FRAME_DIR = "inference/frames_S4_Drill_side"
ANNOTATION_DIR = "datasets/annotations/json_train"
CHECKPOINT = "experiments/checkpoints/model_epoch_15.pth"
NUM_CLASSES = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
print("⚙️ Loading trained model...")
model = SlowFastModel(num_classes=NUM_CLASSES)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint)
model = model.to(DEVICE)
model.eval()

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

# === LOAD FRAMES ===
all_frames = {}
for f in sorted(os.listdir(FRAME_DIR)):
    if not f.endswith(".jpg"):
        continue
    try:
        frame_num = int(os.path.splitext(f)[0])
        img = Image.open(os.path.join(FRAME_DIR, f)).convert("RGB")
        all_frames[frame_num] = transform(img)
    except Exception as e:
        print(f"⚠️ Skipping frame {f}: {e}")

if not all_frames:
    print("🚫 No frames found in folder.")
    exit()

print(f"📦 Loaded {len(all_frames)} frames from '{FRAME_DIR}'\n")

# === LOAD ANNOTATION FILES ===
annotation_files = [f for f in os.listdir(ANNOTATION_DIR) if f.endswith(".json")]
segments = []

for ann_file in annotation_files:
    path = os.path.join(ANNOTATION_DIR, ann_file)
    with open(path, 'r') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                segments.extend(data)
            else:
                segments.append(data)
        except Exception as e:
            print(f"⚠️ Failed to load {ann_file}: {e}")

print(f"📄 Loaded {len(segments)} annotated segments from '{ANNOTATION_DIR}'\n")

# === RUN INFERENCE ===
predicted_segments = []

for idx, seg in enumerate(segments):
    start = seg.get("start_frame")
    end = seg.get("end_frame")
    text = seg.get("text", "unknown")

    if start is None or end is None:
        print(f"⚠️ Skipping segment {idx} (missing start or end frame)")
        continue

    frame_ids = list(range(start, end + 1))

    # Ensure we have at least 32 frames
    while len(frame_ids) < 32:
        frame_ids.append(frame_ids[-1])

    slow_ids = np.linspace(0, len(frame_ids) - 1, 32).astype(int)
    fast_ids = np.linspace(0, len(frame_ids) - 1, 8).astype(int)

    try:
        slow_tensor = torch.stack([all_frames[frame_ids[i]] for i in slow_ids])
        fast_tensor = torch.stack([all_frames[frame_ids[i]] for i in fast_ids])
    except KeyError as e:
        print(f"⚠️ Missing frame for segment {idx}: {e}")
        continue

    x_slow = slow_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)
    x_fast = fast_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x_slow, x_fast)
        pred_class = outputs.argmax(dim=1).item()

    predicted_segments.append({
        "segment_index": idx,
        "start_frame": start,
        "end_frame": end,
        "true_text": text,
        "predicted_class": pred_class,
        "predicted_label": f"Class {pred_class}"
    })

    print(f"Segment {idx}: Frames {start} - {end} → Activity: {text} → Predicted class: {pred_class}")

# === SAVE TO JSON ===
output_path = "inference/predicted_results.json"
with open(output_path, "w") as f:
    json.dump(predicted_segments, f, indent=2)

print(f"\n✅ Prediction results saved to: {output_path}")
