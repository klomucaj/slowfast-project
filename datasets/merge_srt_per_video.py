import os
import pysrt
from collections import defaultdict

# Folder where original detailed .srt files are
INPUT_DIR = "datasets/annotations/annotationsGroup"
# Output folder where merged .srt files will be saved
OUTPUT_DIR = "datasets/annotations/srt_merged"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper: strip suffix after first "." in filename to get video ID
def get_video_key(filename):
    return filename.split(".")[0]  # S1-ADL1_side

# Group files by video ID
srt_groups = defaultdict(list)
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".srt"):
        key = get_video_key(filename)
        srt_groups[key].append(filename)

for video_key, files in srt_groups.items():
    all_subs = []

    for f in files:
        path = os.path.join(INPUT_DIR, f)
        subs = pysrt.open(path)

        # Optional: tag text with source name
        for sub in subs:
            tag = f"[{f.split('.')[-2]}]"  # e.g. 'right_arm'
            sub.text = f"{tag} {sub.text}"
            all_subs.append(sub)

    # Sort by start time
    all_subs.sort(key=lambda s: s.start.ordinal)

    # Write to new merged .srt file
    output_path = os.path.join(OUTPUT_DIR, f"{video_key}.srt")
    pysrt.SubRipFile(all_subs).save(output_path, encoding='utf-8')
    print(f" Merged {len(files)} files → {output_path}")
