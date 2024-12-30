import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time

class TrainUtils:
    """
    Utility class to handle training loops, metrics, and losses for SlowFast networks.
    """
    def __init__(self, model, optimizer, criterion, device, log_dir="runs/train"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = SummaryWriter(log_dir)
        self.start_time = time.time()

    def train_one_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        self.writer.add_scalar("Training Loss", avg_loss, epoch)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        return avg_loss

    def save_model_checkpoint(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"Checkpoint saved at epoch {epoch} to {path}")

    def print_time_elapsed(self):
        print(f"Total time elapsed: {time.time() - self.start_time:.2f} seconds")
