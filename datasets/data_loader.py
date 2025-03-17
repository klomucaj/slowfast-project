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

        # ✅ Check if directories exist
        if not os.path.exists(self.frame_root_dir):
            raise ValueError(f"❌ Error: Frame directory not found: {self.frame_root_dir}")
        if not os.path.exists(self.annotation_dir):
            raise ValueError(f"❌ Error: Annotation directory not found: {self.annotation_dir}")

        # ✅ Load annotation files and create a mapping (video_name -> JSON file)
        self.annotations = {}
        for json_file in os.listdir(self.annotation_dir):
            if json_file.endswith(".json"):
                video_name = json_file.replace(".json", "")
                self.annotations[video_name] = os.path.join(self.annotation_dir, json_file)

        # ✅ Identify valid videos
        self.valid_videos = []
        for video_name, annotation_path in self.annotations.items():
            frame_dir = os.path.join(self.frame_root_dir, video_name)

            # ✅ Ensure frame directory exists
            if not os.path.exists(frame_dir):
                print(f"⚠ Warning: No frame directory found for {video_name}, skipping!")
                continue

            # ✅ Load JSON annotation
            with open(annotation_path, "r") as f:
                annotation_data = json.load(f)

            frame_list = annotation_data.get("frames", [])
            label = annotation_data.get("label", None)

            # ✅ Validate annotation contents
            if not frame_list:
                print(f"⚠ Warning: No frames listed in annotation for {video_name}, skipping!")
                continue
            if label is None:
                print(f"⚠ Warning: Missing label for {video_name}, skipping!")
                continue

            # ✅ Check if frames exist
            valid_frames = [f for f in frame_list if os.path.exists(os.path.join(frame_dir, f))]
            if not valid_frames:
                print(f"⚠ Warning: No valid frames found for {video_name}, skipping!")
                continue

            self.valid_videos.append((video_name, valid_frames, label))

        # ✅ If dataset is empty, raise an error early
        if len(self.valid_videos) == 0:
            raise ValueError("❌ Error: No valid training samples found! Check dataset directories.")

    def __getitem__(self, idx):
        video_name, frame_list, label = self.valid_videos[idx]
        frame_dir = os.path.join(self.frame_root_dir, video_name)

        # ✅ Load and transform frames
        frames = []
        for frame_file in frame_list:
            frame_path = os.path.join(frame_dir, frame_file)
            try:
                image = Image.open(frame_path).convert("RGB")
                frames.append(self.transform(image))
            except Exception as e:
                print(f"⚠ Warning: Failed to load frame {frame_path}: {e}")
                continue

        if not frames:
            raise ValueError(f"❌ Error: No valid frames could be loaded for {video_name}!")

        frames = torch.stack(frames)  # Shape: (T, C, H, W)

        return frames, label

    def __len__(self):
        return len(self.valid_videos)

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])
