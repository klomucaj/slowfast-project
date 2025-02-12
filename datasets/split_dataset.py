import os
import shutil
import random

class SplitDataset:
    def __init__(self, annotation_folders, video_folders, output_base, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.annotation_folders = annotation_folders
        self.video_folders = video_folders
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Define output directories
        self.output_dirs = {
            "srt": {
                "train": os.path.join(output_base, "annotations/srt_train"),
                "val": os.path.join(output_base, "annotations/srt_val"),
                "test": os.path.join(output_base, "annotations/srt_test"),
            },
            "video": {
                "train": os.path.join(output_base, "videos/train_videos"),
                "val": os.path.join(output_base, "videos/val_videos"),
                "test": os.path.join(output_base, "videos/test_videos"),
            }
        }

        # Ensure all output directories exist
        for category in self.output_dirs.values():
            for split in category.values():
                os.makedirs(split, exist_ok=True)

    def collect_files(self, folders, extensions):
        """Collects all files with the given extensions from multiple directories."""
        files = []
        for folder in folders:
            if os.path.exists(folder):
                files.extend(
                    [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(tuple(extensions))]
                )
        return files

    def split_files(self, files):
        """Shuffles and splits files into train, val, and test."""
        random.shuffle(files)
        total = len(files)
        train_split = int(total * self.train_ratio)
        val_split = int(total * self.val_ratio)

        return {
            "train": files[:train_split],
            "val": files[train_split:train_split + val_split],
            "test": files[train_split + val_split:]
        }

    def move_files(self, file_dict, category):
        """Moves files to the appropriate output directory."""
        for split, files in file_dict.items():
            for file in files:
                shutil.copy(file, os.path.join(self.output_dirs[category][split], os.path.basename(file)))

    def process(self):
        """Runs the full dataset splitting pipeline."""
        # Step 1: Collect and split SRT files
        srt_files = self.collect_files(self.annotation_folders, [".srt"])
        srt_splits = self.split_files(srt_files)
        self.move_files(srt_splits, "srt")

        # Step 2: Collect and split Videos
        video_files = self.collect_files(self.video_folders, [".mp4", ".avi", ".mov"])
        video_splits = self.split_files(video_files)
        self.move_files(video_splits, "video")

        print("\n Dataset splitting completed successfully!")
        print(f"Total Annotations: {len(srt_files)} → Train: {len(srt_splits['train'])}, Val: {len(srt_splits['val'])}, Test: {len(srt_splits['test'])}")
        print(f"Total Videos: {len(video_files)} → Train: {len(video_splits['train'])}, Val: {len(video_splits['val'])}, Test: {len(video_splits['test'])}")

# Main execution
if __name__ == "__main__":
    annotation_folders = ["datasets/annotations/S1", "datasets/annotations/S2", "datasets/annotations/S3", "datasets/annotations/S4"]
    video_folders = ["datasets/videos/S1", "datasets/videos/S2", "datasets/videos/S3", "datasets/videos/S4"]
    output_base = "datasets"

    dataset_splitter = SplitDataset(annotation_folders, video_folders, output_base)
    dataset_splitter.process()
