import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.data_loader import FrameDataset
from datasets.models.slowfast_model import SlowFastModel
import yaml

# 🔹 Load YAML configuration
CONFIG_PATH = "configs/slowfast_training.yaml"
with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

# 🔹 Ensure dataset paths are correctly formatted
TRAIN_FRAMES_DIR = os.path.abspath(config["DATA"]["TRAIN_DIR"])
VAL_FRAMES_DIR = os.path.abspath(config["DATA"]["VAL_DIR"])
TRAIN_ANNOTATIONS_DIR = os.path.abspath(config["DATA"]["TRAIN_ANNOTATIONS"])
VAL_ANNOTATIONS_DIR = os.path.abspath(config["DATA"]["VAL_ANNOTATIONS"])
BATCH_SIZE = config["DATA"]["BATCH_SIZE"]
NUM_WORKERS = config["DATA"]["NUM_WORKERS"]
EPOCHS = config["SOLVER"]["MAX_EPOCH"]
LEARNING_RATE = config["SOLVER"]["BASE_LR"]
CHECKPOINT_DIR = os.path.abspath(config["CHECKPOINT"]["SAVE_DIR"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 🔹 Verify dataset paths
def verify_directory(path, name):
    if not os.path.exists(path) or not os.path.isdir(path):
        raise FileNotFoundError(f"Error: {name} directory {path} does not exist or is not a directory.")
    print(f"{name} directory verified: {path}")

def verify_annotations(path, name):
    """Check if annotation directory contains JSON files."""
    if not os.path.exists(path) or not os.path.isdir(path):
        raise FileNotFoundError(f"Error: {name} directory {path} does not exist or is not a directory.")

    json_files = [f for f in os.listdir(path) if f.endswith(".json")]
    if len(json_files) == 0:
        raise ValueError(f"Error: {name} directory {path} does not contain any JSON files!")

    print(f"{name} directory verified with {len(json_files)} JSON files: {path}")

verify_directory(TRAIN_FRAMES_DIR, "Train Frames")
verify_directory(VAL_FRAMES_DIR, "Validation Frames")
verify_annotations(TRAIN_ANNOTATIONS_DIR, "Train Annotations")
verify_annotations(VAL_ANNOTATIONS_DIR, "Validation Annotations")

# 🔹 Load datasets
try:
    print("🔹 Initializing datasets and dataloaders...")
    train_dataset = FrameDataset(TRAIN_FRAMES_DIR, TRAIN_ANNOTATIONS_DIR, transform=None)
    val_dataset = FrameDataset(VAL_FRAMES_DIR, VAL_ANNOTATIONS_DIR, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print("Datasets initialized successfully!")
except Exception as e:
    raise ValueError(f"Error initializing dataset: {e}")

# 🔹 Initialize model
model = SlowFastModel(num_classes=config["MODEL"]["NUM_CLASSES"]).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 🔹 Training function
def train():
    print(f"🚀 Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct, total = 0, 0

        for batch_idx, (frames, labels) in enumerate(train_loader):
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(frames)

            if outputs.shape[0] == 0:
                print(f"Skipping empty batch at index {batch_idx}")
                continue

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total if total > 0 else 0
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss:.4f} - Accuracy: {accuracy:.2f}%")

        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"slowfast_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved: {checkpoint_path}")

# 🔹 Run training
if __name__ == "__main__":
    train()
    print("Training completed successfully!")
