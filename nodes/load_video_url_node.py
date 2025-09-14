import os
import requests
from io import BytesIO
import folder_paths
from comfy.comfy_types import IO, ComfyNodeABC
from comfy_api.latest import InputImpl


def load_video(video_source):
    if video_source.startswith('http'):
        print(f"Downloading video from: {video_source}")
        response = requests.get(video_source)
        response.raise_for_status()

        # Get filename from URL or use a default
        file_name = video_source.split('/')[-1]
        if not file_name or '.' not in file_name:
            file_name = "downloaded_video.mp4"

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

    RETURN_TYPES = (IO.VIDEO,)
    FUNCTION = "load"
    CATEGORY = "image/video"

    def load(self, url_or_path):
        print(f"Loading video from: {url_or_path}")
        video_path, name = load_video(url_or_path)
        return (InputImpl.VideoFromFile(video_path),)


if __name__ == "__main__":
    # Test with a sample video URL (commented out for safety)
    # video_path, name = load_video("https://example.com/sample.mp4")
    # print(f"Loaded video: {name} from {video_path}")
    pass