import os
import json

def srt_to_json(srt_folder, output_json):
    """
    Convert .srt annotation files to a JSON format.
    Args:
        srt_folder (str): Path to folder containing .srt files.
        output_json (str): Output path for the JSON file.
    """
    annotations = []

    for file in os.listdir(srt_folder):
        if file.endswith(".srt"):
            file_path = os.path.join(srt_folder, file)
            with open(file_path, "r") as f:
                lines = f.readlines()

            video_data = {"video": file.replace(".srt", ""), "annotations": []}

            for i in range(0, len(lines), 4):  # SRT has blocks of 4 lines
                start_end = lines[i+1].strip().split(" --> ")
                caption = lines[i+2].strip()
                video_data["annotations"].append({
                    "start_time": start_end[0],
                    "end_time": start_end[1],
                    "action": caption
                })

            annotations.append(video_data)

    # Save as JSON
    with open(output_json, "w") as out_file:
        json.dump(annotations, out_file, indent=4)
    print(f"Annotations saved to {output_json}")

# Example usage
if __name__ == "__main__":
    srt_folder = "../datasets/annotations"
    output_json = "../datasets/annotations/annotations.json"
    srt_to_json(srt_folder, output_json)
