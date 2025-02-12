import os
import json

# Define paths for input (SRT files) and output (JSON files)
srt_folders = {
    "train": "datasets/annotations/srt_train",
    "val": "datasets/annotations/srt_val",
    "test": "datasets/annotations/srt_test"
}

json_folders = {
    "train": "datasets/annotations/json_train",
    "val": "datasets/annotations/json_val",
    "test": "datasets/annotations/json_test"
}

# Ensure output directories exist
for folder in json_folders.values():
    os.makedirs(folder, exist_ok=True)

# Function to convert SRT to JSON
def srt_to_json(srt_file):
    json_data = []
    with open(srt_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    subtitle_block = {}
    for line in lines:
        line = line.strip()
        if line.isdigit():
            if subtitle_block:
                json_data.append(subtitle_block)
                subtitle_block = {}
            subtitle_block["index"] = int(line)
        elif "-->" in line:
            subtitle_block["time"] = line
        elif line:
            subtitle_block.setdefault("text", []).append(line)

    if subtitle_block:
        json_data.append(subtitle_block)

    return json_data

# Process each SRT file and save as JSON
def convert_srt_to_json(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".srt"):
            srt_file = os.path.join(input_folder, filename)
            json_data = srt_to_json(srt_file)

            # Define JSON file path
            json_file = os.path.join(output_folder, filename.replace(".srt", ".json"))

            # Save JSON data
            with open(json_file, "w", encoding="utf-8") as json_out:
                json.dump(json_data, json_out, indent=4, ensure_ascii=False)

            print(f"Converted {srt_file} → {json_file}")

# Run conversion for train, val, and test sets
for split in ["train", "val", "test"]:
    convert_srt_to_json(srt_folders[split], json_folders[split])

print("JSON generation completed successfully!")
