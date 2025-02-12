import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FrameDataset(Dataset):
    """
    Dataset class to load video frames for SlowFast model training.
    """

    def __init__(self, frame_root_dir, annotation_dir, transform=None):
        """
        Args:
            frame_root_dir (str): Path to the directory containing extracted frames.
            annotation_dir (str): Path to the directory containing annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.frame_root_dir = frame_root_dir  # E.g., "datasets/frames/train"
        self.annotation_dir = annotation_dir  # E.g., "datasets/annotations/json_train"
        self.transform = transform if transform else self.default_transform()

        # Get sorted list of video folders
        self.video_folders = sorted(os.listdir(self.frame_root_dir))
        self.annotation_files = sorted(os.listdir(self.annotation_dir))
        self.action_to_index = self._load_action_mapping()

    def _load_action_mapping(self):
        """
        Defines the mapping of action labels to class indices.
        """
        return {"action1": 0, "action2": 1, "action3": 2}  # Update with real action labels

    def _load_annotation(self, annotation_path):
        """
        Loads the annotation file to get the class label.
        """
        with open(annotation_path, 'r') as f:
            return self.action_to_index[f.read().strip()]

    def __getitem__(self, idx):
        """
        Loads the frames and corresponding label for a given index.
        """
        video_folder = self.video_folders[idx]
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])

        # Get list of frames sorted by filename (frame_0001.jpg, frame_0002.jpg, ...)
        frame_dir = os.path.join(self.frame_root_dir, video_folder)
        frame_paths = sorted(
            [os.path.join(frame_dir, frame) for frame in os.listdir(frame_dir) if frame.endswith(('.jpg', '.png'))]
        )

        frames = [self.transform(Image.open(frame)) for frame in frame_paths]
        frames = torch.stack(frames)  # Shape: (T, C, H, W)

        label = self._load_annotation(annotation_path)

        return frames, label  # Shape: (T, C, H, W), label_index

    def __len__(self):
        """
        Returns the total number of video samples.
        """
        return len(self.video_folders)

    def default_transform(self):
        """
        Default transformation: Resize, normalize, convert to tensor.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize frames
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])  # Normalize
        ])
