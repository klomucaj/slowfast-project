import os
import json
import torch
import random
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, RandomHorizontalFlip, RandomResizedCrop, ColorJitter


class FrameDataset(Dataset):
    def __init__(self, frame_root, annotation_root, num_frames=32, transform=None, is_train=True):
        self.frame_root = frame_root
        self.annotation_root = annotation_root
        self.num_frames = num_frames
        self.is_train = is_train

        if transform:
            self.transform = transform
        else:
            base_transforms = [
                Resize(256),
                CenterCrop(224),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ]
            aug_transforms = [
                RandomResizedCrop(224, scale=(0.8, 1.0)),
                RandomHorizontalFlip(),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ]
            self.transform = Compose(aug_transforms + base_transforms) if is_train else Compose(base_transforms)

        self.samples = self._load_samples()
        if not self.samples:
            raise ValueError("Error: No valid training samples found! Check dataset directories.")

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

        total = len(frame_files)
        if self.is_train:
            max_start = max(0, total - self.num_frames)
            clip_start = random.randint(0, max_start)
        else:
            mid = (item["start"] + item["end"]) // 2
            clip_start = max(0, mid - self.num_frames // 2)

        clip_end = min(total, clip_start + self.num_frames)
        clip_start = max(0, clip_end - self.num_frames)
        selected_frames = frame_files[clip_start:clip_end]

        frames = []
        for frame_name in selected_frames:
            img_path = os.path.join(frame_dir, frame_name)
            img = read_image(img_path).float() / 255.0
            img = self.transform(img)
            frames.append(img)

        video_tensor = torch.stack(frames)
        x_fast = video_tensor[::4]
        x_slow = video_tensor

        x_fast = x_fast.permute(1, 0, 2, 3)  # [C, T_fast, H, W]
        x_slow = x_slow.permute(1, 0, 2, 3)  # [C, T_slow, H, W]

        label_tensor = torch.tensor(item["label"])

        return (x_slow, x_fast), label_tensor
 # Debug print every 50 samples
        if idx % 50 == 0:
            print(f"[Dataset] idx: {idx}, sample shape: {sample.shape}, label: {label}")

        return sample, label