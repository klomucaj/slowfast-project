import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

class FrameDataset(Dataset):
    def __init__(self, frame_root, annotation_root, num_frames=32, transform=None):
        self.frame_root = frame_root
        self.annotation_root = annotation_root
        self.num_frames = num_frames

        # Default transform pipeline — note: no ToTensor()
        self.transform = transform if transform else Compose([
            Resize(256),
            CenterCrop(224),
            Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])

        self.samples = self._load_samples()
        if not self.samples:
            raise ValueError(" Error: No valid training samples found! Check dataset directories.")

    def _load_samples(self):
        samples = []
        for fname in os.listdir(self.annotation_root):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.annotation_root, fname)
            with open(path, "r") as f:
                ann_list = json.load(f)

            for ann in ann_list:
                video = ann["video"]
                start = ann["start_frame"]
                end = ann["end_frame"]
                label = ann["label"]

                if (end - start + 1) >= self.num_frames:
                    samples.append({
                        "video": video,
                        "start": start,
                        "end": end,
                        "label": label
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        frame_dir = os.path.join(self.frame_root, item["video"])
        frame_files = sorted(os.listdir(frame_dir))

        # Clip sampling
        start = item["start"]
        end = item["end"]
        mid = (start + end) // 2
        half = self.num_frames // 2
        clip_start = max(0, mid - half)
        clip_end = clip_start + self.num_frames

        if clip_end > len(frame_files):
            clip_end = len(frame_files)
            clip_start = max(0, clip_end - self.num_frames)

        selected_frames = frame_files[clip_start:clip_end]

        frames = []
        for frame_name in selected_frames:
            img_path = os.path.join(frame_dir, frame_name)
            img = read_image(img_path).float() / 255.0  # Already a torch.Tensor in [0,1]
            img = self.transform(img)
            frames.append(img)

        video_tensor = torch.stack(frames)  # Shape: [T, C, H, W]
        label_tensor = torch.tensor(item["label"])

        return video_tensor, label_tensor
