import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FrameDataset(Dataset):
    def __init__(self, frame_root_dir, annotation_dir, transform=None):
        """
        Args:
            frame_root_dir (str): Path to the directory containing extracted frames.
            annotation_dir (str): Path to the directory containing JSON annotations.
            transform (callable, optional): Optional transform to be applied to frames.
        """
        self.frame_root_dir = frame_root_dir
        self.annotation_dir = annotation_dir
        self.transform = transform if transform else self.default_transform()

        # 🔹 Get list of all video folders inside frame directory
        self.video_folders = sorted(os.listdir(self.frame_root_dir))

        # 🔹 Load annotation files and create a mapping (video_name -> JSON file)
        self.annotations = {}
        for json_file in os.listdir(self.annotation_dir):
            if json_file.endswith(".json"):
                video_name = json_file.replace(".json", "")
                self.annotations[video_name] = os.path.join(self.annotation_dir, json_file)

        # 🔹 Keep only frame folders that have a matching JSON annotation
        self.valid_videos = []
        for video_folder in self.video_folders:
            if video_folder in self.annotations:
                frame_dir = os.path.join(self.frame_root_dir, video_folder)
                frame_files = [f for f in os.listdir(frame_dir) if f.endswith((".jpg", ".png"))]
                
                # Ensure there are frames in the folder
                if len(frame_files) > 0:
                    self.valid_videos.append(video_folder)
                else:
                    print(f"Warning: {video_folder} has a matching annotation but no frames, skipping!")

        # Log missing annotations
        missing_annotations = set(self.video_folders) - set(self.annotations.keys())
        if missing_annotations:
            print(f"Warning: {len(missing_annotations)} frame directories have no matching annotation and will be skipped.")

        # 🔹 If dataset is empty, raise an error early
        if len(self.valid_videos) == 0:
            raise ValueError("Error: No valid training samples found! Check dataset directories.")

    def __getitem__(self, idx):
        video_folder = self.valid_videos[idx]
        annotation_path = self.annotations[video_folder]

        # 🔹 Load frames from the corresponding video folder
        frame_dir = os.path.join(self.frame_root_dir, video_folder)
        frame_paths = sorted(
            [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith((".jpg", ".png"))]
        )

        # 🔹 Load all frames and apply transforms
        frames = [self.transform(Image.open(frame)) for frame in frame_paths]
        frames = torch.stack(frames)  # Shape: (T, C, H, W)

        # 🔹 Load JSON annotation
        with open(annotation_path, "r") as f:
            annotation_data = json.load(f)

        label = annotation_data["label"]  # Adjust based on JSON format

        return frames, label

    def __len__(self):
        return len(self.valid_videos)

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])
