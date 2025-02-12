import torch
from torch.utils.data import DataLoader
from datasets.data_loader import VideoDataset
from datasets.models.slowfast_model import SlowFastModel
from datasets.utils.eval_utils import inference
import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SlowFast Testing/Inference Script")
    parser.add_argument('--config', type=str, default='configs/slowfast_inference.yaml', help='Path to config file')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("Loading test dataset...")
    test_dataset = VideoDataset(
        video_folder=config['data']['test_videos'],
        annotation_file=config['data']['test_annotations']
    )
    test_loader = DataLoader(test_dataset, batch_size=config['test']['batch_size'], shuffle=False)

    # Load model
    print("Loading model...")
    model = SlowFastModel(num_classes=config['model']['num_classes'])
    model.load_state_dict(torch.load(config['test']['checkpoint_path']))
    model = model.to(device)
    model.eval()

    # Run inference
    print("Running inference...")
    predictions = inference(model, test_loader, device)
    print("Predictions completed. Results saved.")

if __name__ == "__main__":
    main()
