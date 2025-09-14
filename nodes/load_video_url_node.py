import os
import requests
from io import BytesIO
import folder_paths
import cv2
import numpy as np
import torch
from comfy.comfy_types import ComfyNodeABC


def load_video(video_source):
    if video_source.startswith('http'):
        print(f"Downloading video from: {video_source}")
        response = requests.get(video_source)
        response.raise_for_status()

        # Get filename from URL, removing query parameters
        url_path = video_source.split('?')[0]  # Remove query parameters
        file_name = url_path.split('/')[-1]

        # If no valid filename found, create one
        if not file_name or '.' not in file_name:
            file_name = "downloaded_video.mp4"

        # Clean filename of invalid characters for Windows
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            file_name = file_name.replace(char, '_')

        # Save to temp file
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file_name)

        with open(temp_path, 'wb') as f:
            f.write(response.content)

        return temp_path, file_name
    else:
        # Local file path
        if os.path.exists(video_source):
            file_name = os.path.basename(video_source)
            return video_source, file_name
        else:
            raise FileNotFoundError(f"Video file not found: {video_source}")


class LoadVideoByUrlOrPath(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_or_path": ("STRING", {"multiline": True, "dynamicPrompts": False})
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "load"
    CATEGORY = "image/video"

    def extract_frames(self, video_path):
        """Extract frames from video and return as torch tensor along with fps"""
        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        frame_count = 0

        while True:
            ret, frame = video_cap.read()
            if not ret:
                break

            # Convert BGR to RGB (opencv loads as BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to float32 and normalize to [0,1] range
            frame = np.array(frame, dtype=np.float32) / 255.0
            frames.append(frame)
            frame_count += 1

        video_cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")

        # Convert to torch tensor with shape [num_frames, height, width, channels]
        frames_tensor = torch.from_numpy(np.stack(frames))

        return frames_tensor, float(fps)

    def load(self, url_or_path):
        print(f"Loading video from: {url_or_path}")
        video_path, name = load_video(url_or_path)
        frames, fps = self.extract_frames(video_path)
        return (frames, fps)


if __name__ == "__main__":
    # Test with a sample video URL (commented out for safety)
    # video_path, name = load_video("https://example.com/sample.mp4")
    # print(f"Loaded video: {name} from {video_path}")
    pass