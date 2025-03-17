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

# Function to check if folders exist and are valid
def validate_folder(folder_path):
    """Check if the folder exists and is a valid directory."""
    if not os.path.exists(folder_path):
        print(f" Error: Folder {folder_path} does not exist.")
        return False
    if not os.path.isdir(folder_path):
        print(f" Error: {folder_path} is not a valid directory.")
        return False
    return True

# Function to fix file permissions
def fix_permissions(json_folder):
    """Ensure correct read/write permissions."""
    for root, _, files in os.walk(json_folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                os.chmod(file_path, 0o777)  # Full permissions
            except Exception as e:
                print(f" Warning: Failed to change permissions for {file_path}: {e}")

# Function to parse an SRT file into a structured JSON format
def parse_srt(srt_file):
    """Convert an SRT file into a structured JSON format."""
    annotations = []
    
    try:
        with open(srt_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except Exception as e:
        print(f" Error reading {srt_file}: {e}")
        return []

    entry = {}
    for line in lines:
        line = line.strip()
        if line.isdigit():  # Subtitle index (ignored)
            continue
        elif "-->" in line:  # Timestamp line
            start_time, end_time = line.split(" --> ")
            entry["start_time"] = start_time
            entry["end_time"] = end_time
        elif line:  # Subtitle text
            entry.setdefault("text", []).append(line)
        else:  # Empty line signals a new subtitle block
            if entry:
                annotations.append(entry)
                entry = {}

    if entry:
        annotations.append(entry)  # Add the last subtitle block

    return annotations

# Function to convert all SRT files in a folder to JSON
def convert_srt_to_json(input_folder, output_folder):
    """Convert SRT files to JSON and save them in the output folder."""
    if not validate_folder(input_folder):
        return

    for filename in os.listdir(input_folder):
        if filename.endswith(".srt"):
            srt_file = os.path.join(input_folder, filename)
            annotations = parse_srt(srt_file)

            # Define JSON file path
            json_filename = filename.replace(".srt", ".json")
            json_file = os.path.join(output_folder, json_filename)

            try:
                with open(json_file, "w", encoding="utf-8") as json_out:
                    json.dump(annotations, json_out, indent=4, ensure_ascii=False)
                print(f" Successfully converted: {srt_file} → {json_file}")
            except Exception as e:
                print(f" Error writing {json_file}: {e}")

# Process SRT files and generate JSON annotations for train, val, and test
for split in ["train", "val", "test"]:
    convert_srt_to_json(srt_folders[split], json_folders[split])
    fix_permissions(json_folders[split])

print(" JSON annotation generation completed successfully!")
