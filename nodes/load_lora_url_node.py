import os
import requests
from io import BytesIO
import hashlib
import folder_paths
import json
import time
import shutil
from pathlib import Path


# This ComfyUI node allows you to load LoRA files from URLs or local paths.
# It downloads the files if they are not already present in the specified folder.
# It also allows you to stack multiple LoRA files together with different strengths.
# Additionally, it manages disk space by tracking usage and removing least recently used files when space is low.
class LoadLoraByUrlOrPath:
    def __init__(self):
        # Initialize lora folder path
        self.lora_folder = folder_paths.get_folder_paths("loras")[0]
        # Ensure history file exists
        self._ensure_history_file()

    def _ensure_history_file(self):
        """Initialize the usage history file if it doesn't exist"""
        history_path = self._get_history_path()
        if not os.path.exists(history_path):
            self._save_history({})

    def _get_history_path(self):
        """Get the path to the usage history file"""
        return os.path.join(self.lora_folder, ".lora_usage_history.json")

    def _load_history(self):
        """Load the LoRA usage history from file"""
        history_path = self._get_history_path()
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_history(self, history):
        """Save the LoRA usage history to file"""
        history_path = self._get_history_path()
        with open(history_path, 'w') as f:
            json.dump(history, f)

    def _update_lora_usage(self, lora_name):
        """Update the last usage timestamp for a LoRA file"""
        history = self._load_history()
        history[lora_name] = time.time()
        self._save_history(history)

    def _check_disk_space(self):
        """Check available disk space in the LoRA folder"""
        try:
            total, used, free = shutil.disk_usage(self.lora_folder)
            return free
        except Exception as e:
            print(f"Error checking disk space: {e}")
            return 0

    def _delete_least_recently_used_lora(self):
        """Delete the least recently used LoRA file to free up space"""
        history = self._load_history()

        # Get all existing LoRA files (excluding hidden files and the history file)
        lora_files = [f for f in os.listdir(self.lora_folder)
                      if os.path.isfile(os.path.join(self.lora_folder, f))
                      and not f.startswith('.')]

        # Filter history to only include existing files
        valid_history = {k: v for k, v in history.items() if k in lora_files}

        # If no history or no files, nothing to delete
        if not valid_history or not lora_files:
            print("No LoRA files available for deletion")
            return False

        # Find least recently used LoRA
        least_recent_lora = min(valid_history.items(), key=lambda x: x[1])
        least_recent_file = least_recent_lora[0]
        last_used_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                       time.localtime(least_recent_lora[1]))

        # Delete the file
        try:
            lora_path = os.path.join(self.lora_folder, least_recent_file)
            file_size = os.path.getsize(lora_path) / (1024 * 1024)  # Size in MB

            os.remove(lora_path)
            print(f"Deleted least recently used LoRA: {least_recent_file} "
                  f"({file_size:.2f} MB, last used: {last_used_time})")

            # Update history by removing the deleted file
            history.pop(least_recent_file)
            self._save_history(history)
            return True
        except Exception as e:
            print(f"Error deleting LoRA file: {e}")
            return False

    @classmethod
    def INPUT_TYPES(s):
        max_lora_num = 10
        inputs = {
            "required": {
                "toggle": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled"}),
                "mode": (["simple", "advanced"],),
                "num_loras": ("INT", {"default": 1, "min": 1, "max": max_lora_num}),
            },
            "optional": {
                "optional_lora_stack": ("LORA_STACK",),
            },
        }

        for i in range(1, max_lora_num + 1):
            inputs["optional"][f"lora_{i}_url"] = ("STRING", {"default": "", "multiline": True,
                                                              "show_if": {"num_loras": {"greater_or_equal": i}}})
            inputs["optional"][f"lora_{i}_strength"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                          "show_if": {"mode": "simple", "num_loras": {"greater_or_equal": i}}})
            inputs["optional"][f"lora_{i}_model_strength"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                          "show_if": {"mode": "advanced", "num_loras": {"greater_or_equal": i}}})
            inputs["optional"][f"lora_{i}_clip_strength"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                          "show_if": {"mode": "advanced", "num_loras": {"greater_or_equal": i}}})

        return inputs

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "load_and_stack"

    CATEGORY = "EasyUse/Loaders"

    def download_lora(self, url):
        """Download a LoRA file from URL or copy from local path with disk space management"""
        # Get filename from url or generate one from hash if no filename is present
        try:
            # Try to get the filename from the URL
            if url.startswith('http'):
                filename = os.path.basename(url.split('?')[0])
                # If no extension or no filename, use hash
                if not filename or '.' not in filename:
                    filename = f"{hashlib.md5(url.encode()).hexdigest()}.safetensors"
            else:
                # Local path
                filename = os.path.basename(url)

            # Check if file already exists
            full_path = os.path.join(self.lora_folder, filename)
            if os.path.exists(full_path):
                print(f"LoRA file {filename} already exists, skipping download")
                # Update usage history for existing file
                self._update_lora_usage(filename)
                return filename

            # Check available disk space before downloading
            MIN_FREE_SPACE = 2 * 1024 * 1024 * 1024  # 2GB in bytes
            free_space = self._check_disk_space()

            # If space is less than 2GB, try to free up space
            if free_space < MIN_FREE_SPACE:
                print(f"Low disk space: {free_space / (1024 * 1024 * 1024):.2f}GB available. Need at least 2GB.")

                # Try to delete least recently used LoRAs until enough space is freed
                while free_space < MIN_FREE_SPACE:
                    deleted = self._delete_least_recently_used_lora()
                    if not deleted:
                        print("Could not free up enough space for new LoRA download")
                        return None
                    # Recheck space after deletion
                    free_space = self._check_disk_space()
                    if free_space >= MIN_FREE_SPACE:
                        print(f"Freed up space. Now {free_space / (1024 * 1024 * 1024):.2f}GB available")
                        break

            # Download the file
            if url.startswith('http'):
                print(f"Downloading LoRA from {url}")
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an exception for HTTP errors

                # Get file size from headers if available
                file_size = int(response.headers.get('content-length', 0)) / (
                            1024 * 1024) if 'content-length' in response.headers else "unknown"
                print(f"File size: {file_size} MB" if isinstance(file_size, (int, float)) else "File size: unknown")

                # Save the file
                with open(full_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded LoRA to {full_path}")

                # Update usage history for the new file
                self._update_lora_usage(filename)
                return filename
            else:
                # Copy local file if it's a valid path
                if os.path.exists(url):
                    import shutil
                    shutil.copy2(url, full_path)
                    print(f"Copied LoRA from {url} to {full_path}")

                    # Update usage history for the new file
                    self._update_lora_usage(filename)
                    return filename
                else:
                    print(f"Local LoRA file {url} does not exist")
                    return None

        except Exception as e:
            print(f"Error downloading LoRA: {e}")
            return None

    def load_and_stack(self, toggle, mode, num_loras, optional_lora_stack=None, **kwargs):
        """Load and stack LoRA files with usage tracking"""
        if (toggle in [False, None, "False"]):
            return (None,)

        loras = []

        # Import Stack values
        if optional_lora_stack is not None:
            loras.extend([l for l in optional_lora_stack if l[0] != "None"])

            # Update usage timestamps for existing stack items
            for lora_item in loras:
                if lora_item[0] != "None":
                    self._update_lora_usage(lora_item[0])

        # Process each LoRA
        for i in range(1, num_loras + 1):
            lora_url = kwargs.get(f"lora_{i}_url", "")

            if not lora_url:
                continue

            # Download/copy the LoRA file with disk space management
            lora_name = self.download_lora(lora_url)

            if not lora_name:
                continue

            # Add the LoRA to the stack
            if mode == "simple":
                lora_strength = float(kwargs.get(f"lora_{i}_strength", 1.0))
                loras.append((lora_name, lora_strength, lora_strength))
            elif mode == "advanced":
                model_strength = float(kwargs.get(f"lora_{i}_model_strength", 1.0))
                clip_strength = float(kwargs.get(f"lora_{i}_clip_strength", 1.0))
                loras.append((lora_name, model_strength, clip_strength))

        # Refresh the lora list to include newly downloaded files
        folder_paths.get_filename_list("loras")

        return (loras,)