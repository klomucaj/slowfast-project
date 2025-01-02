import cv2
import numpy as np
import os
import json

def analyze_movements(video_path, frame_dir, output_json):
    """
    Analyze movements in a video using optical flow and save annotated data.

    Args:
        video_path (str): Path to the input video file.
        frame_dir (str): Directory to save extracted frames.
        output_json (str): Path to save the annotated JSON file.
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    movement_data = []

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Calculate magnitude and angle of flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Threshold to detect significant motion
        motion_mask = magnitude > 2  # Adjust threshold as needed

        if np.sum(motion_mask) > 0:
            # Describe movement based on the angle
            avg_angle = np.mean(angle[motion_mask])
            direction = describe_motion(avg_angle)

            # Save movement annotation
            movement_data.append({
                "frame_number": frame_count,
                "timestamp": round(frame_count / cap.get(cv2.CAP_PROP_FPS), 3),
                "movement": direction
            })

        # Save frame for reference
        frame_path = os.path.join(frame_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, next_frame)

        # Update for next iteration
        prev_gray = next_gray
        frame_count += 1

    cap.release()

    # Save movement data to JSON
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(movement_data, json_file, indent=4)
    print(f"Movements saved to {output_json}")


def describe_motion(angle):
    """
    Describe motion direction based on the angle in radians.

    Args:
        angle (float): Angle of motion in radians.

    Returns:
        str: Description of the motion direction.
    """
    if 0 <= angle < np.pi / 4 or 7 * np.pi / 4 <= angle < 2 * np.pi:
        return "Rightward movement"
    elif np.pi / 4 <= angle < 3 * np.pi / 4:
        return "Upward movement"
    elif 3 * np.pi / 4 <= angle < 5 * np.pi / 4:
        return "Leftward movement"
    else:
        return "Downward movement"


# Example usage
if __name__ == "__main__":
    video_path = "C:/Users/TINACOM/Desktop/VideoTraining/datasets/annotations/srt_train/S1-ADL1_side.avi"
    frame_dir = "C:/Users/TINACOM/Desktop/VideoTraining/frames/S1-ADL1_side"
    output_json = "C:/Users/TINACOM/Desktop/VideoTraining/frames/S1-ADL1_side_movements.json"

    analyze_movements(video_path, frame_dir, output_json)
