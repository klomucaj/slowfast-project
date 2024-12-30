import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.data_loader import VideoDataset
from datasets.models.slowfast_model import SlowFastModel
from datasets.utils.train_utils import train_one_epoch
from datasets.utils.checkpoint import save_checkpoint
import yaml
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SlowFast Training Script")
    parser.add_argument('--config', type=str, default='configs/slowfast_training.yaml', help='Path to config file')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    print("Loading datasets...")
    train_dataset = VideoDataset(
        video_folder=config['data']['train_videos'],
        annotation_file=config['data']['train_annotations'],
        transform=config['data']['transform']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    
    # Model setup
    print("Initializing model...")
    model = SlowFastModel(num_classes=config['model']['num_classes'])
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    
    # Training loop
    print("Starting training...")
    for epoch in range(config['train']['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{config['train']['epochs']}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % config['train']['save_freq'] == 0:
            save_checkpoint(model, optimizer, epoch, config['train']['checkpoint_path'])

    print("Training complete.")

if __name__ == "__main__":
    main()
