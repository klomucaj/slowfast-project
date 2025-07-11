import os
import json
import pysrt

# === CONFIG ===
SRT_ROOT = "datasets/annotations"
OUTPUT_ROOT = "datasets/annotations"
FPS = 1  # Match this to your frame extraction rate

def time_to_frame(srt_time, fps):
    return int(srt_time.hours * 3600 * fps +
               srt_time.minutes * 60 * fps +
               srt_time.seconds * fps +
               srt_time.milliseconds * fps / 1000)

def convert_split(split):
    srt_dir = os.path.join(SRT_ROOT, f"srt_{split}")
    json_dir = os.path.join(OUTPUT_ROOT, f"json_{split}")
    os.makedirs(json_dir, exist_ok=True)

    for srt_file in os.listdir(srt_dir):
        if not srt_file.endswith(".srt"):
            continue

        video_id = os.path.splitext(srt_file)[0]
        subs = pysrt.open(os.path.join(srt_dir, srt_file))
        annotations = []

        for sub in subs:
            start = time_to_frame(sub.start, FPS)
            end = time_to_frame(sub.end, FPS)
            label = 0  # Default label (can be replaced later)
            text = sub.text.strip()

            annotations.append({
                "video": video_id,
                "start_frame": start,
                "end_frame": end,
                "label": label,
                "text": text
            })

        with open(os.path.join(json_dir, f"{video_id}.json"), "w") as f:
            json.dump(annotations, f, indent=2)
        print(f" Created: {json_dir}/{video_id}.json")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        convert_split(split)
