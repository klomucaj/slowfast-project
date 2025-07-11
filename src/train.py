# train.py
from collections import Counter
from typing import Dict
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.models.slowfast_model import SlowFastModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.data_loader import FrameDataset

# === Load Config ===
with open("configs/slowfast_training.yaml", "r") as f:
    config = yaml.safe_load(f)

# === Config Values ===
TRAIN_FRAMES_DIR = config["DATA"]["TRAIN_DIR"]
TRAIN_ANNOTATIONS_DIR = config["DATA"]["TRAIN_ANNOTATIONS"]
VAL_FRAMES_DIR = config["DATA"]["VAL_DIR"]
VAL_ANNOTATIONS_DIR = config["DATA"]["VAL_ANNOTATIONS"]
BATCH_SIZE = config["DATA"]["BATCH_SIZE"]
NUM_WORKERS = config["DATA"]["NUM_WORKERS"]
NUM_EPOCHS = config["SOLVER"]["MAX_EPOCH"]
LEARNING_RATE = config["SOLVER"]["BASE_LR"]
NUM_CLASSES = config["MODEL"]["NUM_CLASSES"]
SAVE_DIR = config["CHECKPOINT"]["SAVE_DIR"]
NUM_FRAMES = config["DATA"]["INPUT"]["NUM_FRAMES"]

DEVICE = torch.device("cpu")

print(f"Train Frames directory verified: {TRAIN_FRAMES_DIR}")
print(f"Train Annotations directory verified: {TRAIN_ANNOTATIONS_DIR}")

# === Dataset & Dataloaders ===
print("Initializing dataset and dataloader...")
train_dataset = FrameDataset(TRAIN_FRAMES_DIR, TRAIN_ANNOTATIONS_DIR, NUM_FRAMES, is_train=True)
val_dataset = FrameDataset(VAL_FRAMES_DIR, VAL_ANNOTATIONS_DIR, NUM_FRAMES, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# === Class Weights Calculation ===
print("Calculating class weights...")
all_labels = [int(label) for ((_, _), label) in train_dataset]
label_counts: Dict[int, int] = dict(Counter(all_labels))
total_samples = len(train_dataset)
weights = [total_samples / (NUM_CLASSES * label_counts.get(i, 1)) for i in range(NUM_CLASSES)]
print("Class Weights:", weights)

# === Model, Loss, Optimizer ===
model = SlowFastModel(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).float().to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Scheduler ===
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose="INFO")

os.makedirs(SAVE_DIR, exist_ok=True)
print(" Starting training...")

# === Training Loop ===
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, ((x_slow, x_fast), labels) in enumerate(train_loader):
        x_slow = x_slow.to(DEVICE)
        x_fast = x_fast.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x_slow, x_fast)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 10 == 0:
            print(f"[Epoch {epoch+1}] Batch {batch_idx}, Loss: {loss.item():.4f}")

    train_accuracy = correct / total * 100
    print(f"[Epoch {epoch+1}] Training Accuracy: {train_accuracy:.2f}%")

    # === Validation ===
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for (x_slow, x_fast), labels in val_loader:
            x_slow = x_slow.to(DEVICE)
            x_fast = x_fast.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(x_slow, x_fast)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total * 100
    print(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

    scheduler.step(avg_val_loss)

    # === Save Model ===
    checkpoint_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f" Saved checkpoint: {checkpoint_path}")