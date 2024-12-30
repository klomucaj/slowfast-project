import torch
from torch.utils.data import DataLoader
from datasets.data_loader import VideoDataset
from datasets.models.slowfast_model import SlowFastModel
from datasets.utils.eval_utils import validate
import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SlowFast Validation Script")
    parser.add_argument('--config', type=str, default='configs/slowfast_validation.yaml', help='Path to config file')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("Loading validation dataset...")
    val_dataset = VideoDataset(
        video_folder=config['data']['val_videos'],
        annotation_file=config['data']['val_annotations']
    )
    val_loader = DataLoader(val_dataset, batch_size=config['validation']['batch_size'], shuffle=False)

    # Load model
    print("Loading model...")
    model = SlowFastModel(num_classes=config['model']['num_classes'])
    model.load_state_dict(torch.load(config['validation']['checkpoint_path']))
    model = model.to(device)
    model.eval()

    # Run validation
    print("Validating model...")
    val_loss, val_acc = validate(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

if __name__ == "__main__":
    main()
