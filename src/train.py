import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from datasets.data_loader import FrameDataset

# Add root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Load Config ===
with open("configs/slowfast_training.yaml", "r") as f:
    config = yaml.safe_load(f)

# Paths from config
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Train Frames directory verified: {TRAIN_FRAMES_DIR}")
print(f"Train Annotations directory verified: {TRAIN_ANNOTATIONS_DIR}")

# === Init Datasets ===
print("Initializing dataset and dataloader...")
train_dataset = FrameDataset(
    frame_root=TRAIN_FRAMES_DIR,
    annotation_root=TRAIN_ANNOTATIONS_DIR,
    num_frames=NUM_FRAMES
)

val_dataset = FrameDataset(
    frame_root=VAL_FRAMES_DIR,
    annotation_root=VAL_ANNOTATIONS_DIR,
    num_frames=NUM_FRAMES
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# === Lightweight 3D Conv Dummy Model ===
class Dummy3DConv(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] → [B, C, T, H, W]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = Dummy3DConv(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

os.makedirs(SAVE_DIR, exist_ok=True)
print("Starting training...")

# === Training Loop ===
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for batch_idx, (video, label) in enumerate(train_loader):
        video, label = video.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(video)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"[Epoch {epoch+1}] Batch {batch_idx}, Loss: {loss.item():.4f}")

    # === Validation Loop ===
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_video, val_label in val_loader:
            val_video, val_label = val_video.to(DEVICE), val_label.to(DEVICE)
            outputs = model(val_video)
            loss = criterion(outputs, val_label)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.4f}")

    # === Save Checkpoint ===
    checkpoint_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
