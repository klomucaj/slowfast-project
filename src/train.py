import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.data_loader import FrameDataset  # 
from datasets.models.slowfast_model import SlowFastModel

# Paths
TRAIN_FRAMES_DIR = "datasets/frames/train"
VAL_FRAMES_DIR = "datasets/frames/val"
TRAIN_ANNOTATIONS = "datasets/annotations/json_train"
VAL_ANNOTATIONS = "datasets/annotations/json_val"

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataset = FrameDataset(TRAIN_FRAMES_DIR, TRAIN_ANNOTATIONS)
val_dataset = FrameDataset(VAL_FRAMES_DIR, VAL_ANNOTATIONS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Initialize model
model = SlowFastModel(num_classes=10).to(DEVICE)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for frames, labels in train_loader:
        frames, labels = frames.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(frames)  
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    # Validation Step
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for frames, labels in val_loader:
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

print("Training Complete!")
