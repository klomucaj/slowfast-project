import os
import shutil
import re

class VideoAnnotationSplitter:
    def __init__(self, video_folder, annotation_folder, output_base, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.video_folder = video_folder
        self.annotation_folder = annotation_folder
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Define output directories
        self.output_dirs = {
            "video": {
                "train": os.path.join(output_base, "videos/train_videos"),
                "val": os.path.join(output_base, "videos/val_videos"),
                "test": os.path.join(output_base, "videos/test_videos"),
            },
            "srt": {
                "train": os.path.join(output_base, "annotations/srt_train"),
                "val": os.path.join(output_base, "annotations/srt_val"),
                "test": os.path.join(output_base, "annotations/srt_test"),
            }
        }

        # Create directories if they don't exist
        for category in self.output_dirs.values():
            for split in category.values():
                os.makedirs(split, exist_ok=True)

    def collect_files(self, folder, extensions):
        """Collects all files with given extensions from the specified folder."""
        files = []
        if os.path.exists(folder):
            files.extend(
                [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(tuple(extensions))]
            )
        return files

    def split_files(self, files):
        """Splits files into train, val, and test sets based on predefined ratios."""
        total = len(files)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        return {
            "train": files[:train_end],  
            "val": files[train_end:val_end],  
            "test": files[val_end:]  
        }

    def move_files(self, file_dict, category):
        """Moves files to the corresponding output directory."""
        for split, files in file_dict.items():
            for file in files:
                shutil.copy(file, os.path.join(self.output_dirs[category][split], os.path.basename(file)))

    def extract_group(self, filename):
        """Extracts group identifier (S1, S2, S3, S4) from filename."""
        match = re.search(r'(S[1-4])', filename)
        return match.group(1) if match else None

    def split_annotations(self):
        """Splits annotations based on already-split videos."""
        annotation_files = self.collect_files(self.annotation_folder, [".srt"])
        annotation_splits = {"train": [], "val": [], "test": []}

        # Check video files inside train/val/test folders
        for split in ["train", "val", "test"]:
            split_video_files = self.collect_files(self.output_dirs["video"][split], [".mp4", ".avi", ".mov"])
            video_names = {os.path.basename(v).rsplit('.', 1)[0] for v in split_video_files}  # Get names without extension

            for annotation in annotation_files:
                annotation_name = os.path.basename(annotation).rsplit('.', 1)[0]  # Get name without extension
                
                # Check if the annotation matches any video inside the split
                if any(video_name in annotation_name for video_name in video_names):
                    annotation_splits[split].append(annotation)

        self.move_files(annotation_splits, "srt")

    def process(self):
        """Executes the full dataset splitting process."""
        
        # Step 1: Collect and split videos first
        video_files = self.collect_files(self.video_folder, [".mp4", ".avi", ".mov"])
        video_splits = self.split_files(video_files)
        self.move_files(video_splits, "video")

        # Step 2: Split annotations based on existing videos in each split
        self.split_annotations()

        # Print Summary
        print("\n Dataset splitting completed successfully!")
        print(f" Total Videos: {len(video_files)} → Train: {len(video_splits['train'])}, Val: {len(video_splits['val'])}, Test: {len(video_splits['test'])}")
        print(f" Annotations split based on videos inside each folder.")

# Main execution
if __name__ == "__main__":
    video_folder = "datasets/videos/videosGroup"
    annotation_folder = "datasets/annotations/annotationsGroup"
    output_base = "datasets"

    splitter = VideoAnnotationSplitter(video_folder, annotation_folder, output_base)
    splitter.process()
