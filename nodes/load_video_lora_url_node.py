import os
import requests
from io import BytesIO
import hashlib
import folder_paths
import json
import time
import shutil
from pathlib import Path
from nodes import LoraLoader  # Import LoraLoader for applying LoRAs


# This ComfyUI node allows you to load Video LoRA files from URLs or local paths
# and apply them to a model and clip.
# It downloads the files if they are not already present in the specified folder.
# It also manages disk space by tracking usage and removing least recently used files when space is low.
class LoadVideoLoraByUrlOrPath:
    def __init__(self):
        # Initialize lora folder path
        self.lora_folder = folder_paths.get_folder_paths("loras")[0]
        # Ensure history file exists
        self._ensure_history_file()
        # Instantiate a LoraLoader for applying LoRAs
        self.lora_loader = LoraLoader()

    def _ensure_history_file(self):
        """Initialize the usage history file if it doesn't exist"""
        history_path = self._get_history_path()
        if not os.path.exists(history_path):
            self._save_history({})

    def _get_history_path(self):
        """Get the path to the usage history file"""
        return os.path.join(self.lora_folder, ".lora_usage_history.json")  # Shared history

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

    def _get_actual_used_space(self):
        """Calculate actual used space in the LoRA folder using du command"""
        import subprocess

        # Use du -sb for bytes output (more reliable than -sh for parsing)
        result = subprocess.run(
            ['du', '-sb', self.lora_folder],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Output format: "12345678\t/path/to/folder"
        size_str = result.stdout.split()[0]
        total_size = int(size_str)
        print(f"du command result: {total_size / (1024 ** 3):.2f}GB for {self.lora_folder}")
        return total_size

    def _check_disk_space(self):
        """Check available disk space in the LoRA folder
        Returns: tuple (free_space_bytes, is_reliable)
        - free_space_bytes: available space in bytes
        - is_reliable: True if disk reading is trustworthy, False if unreliable
        """
        try:
            total, used, free = shutil.disk_usage(self.lora_folder)
            free_gb = free / (1024 ** 3)

            # If free space > 200 GB, it's unreliable (network volume issue)
            if free_gb > 200:
                print(f"Unreliable disk reading detected ({free_gb:.1f} GB free). Using calculated space.")
                # Use hardcoded max of 200 GB
                MAX_DISK_SPACE_GB = 200
                actual_used_bytes = self._get_actual_used_space()
                actual_used_gb = actual_used_bytes / (1024 ** 3)
                free_space = (MAX_DISK_SPACE_GB * 1024 ** 3) - actual_used_bytes
                print(f"Actual used space: {actual_used_gb:.2f} GB, Calculated free: {free_space / (1024 ** 3):.2f} GB")
                return (free_space, False)  # False = unreliable system reading
            else:
                # Trust the system reading
                print(f"Reliable disk reading: {free_gb:.2f} GB free")
                return (free, True)  # True = reliable system reading
        except Exception as e:
            print(f"Error checking disk space: {e}")
            return (0, True)

    def _delete_least_recently_used_lora(self):
        """Delete the least recently used LoRA file to free up space"""
        history = self._load_history()

        lora_files = [f for f in os.listdir(self.lora_folder)
                      if os.path.isfile(os.path.join(self.lora_folder, f))
                      and not f.startswith('.')
                      and f != os.path.basename(self._get_history_path())]  # Exclude history file itself

        valid_history = {k: v for k, v in history.items() if k in lora_files}

        if not valid_history or not lora_files:
            print("No LoRA files available for deletion based on history or existing files.")
            return False

        least_recent_lora_entry = min(valid_history.items(), key=lambda x: x[1])
        least_recent_file = least_recent_lora_entry[0]

        # Ensure the file still exists before attempting deletion
        lora_path_to_delete = os.path.join(self.lora_folder, least_recent_file)
        if not os.path.exists(lora_path_to_delete):
            print(f"Skipping deletion of {least_recent_file}, file not found (already deleted or moved).")
            # Remove from history if it's not found
            if least_recent_file in history:
                history.pop(least_recent_file)
                self._save_history(history)
            return False  # Indicate that no actual deletion occurred that would free space

        last_used_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(least_recent_lora_entry[1]))

        try:
            file_size_mb = os.path.getsize(lora_path_to_delete) / (1024 * 1024)
            os.remove(lora_path_to_delete)
            print(f"Deleted least recently used LoRA: {least_recent_file} "
                  f"({file_size_mb:.2f} MB, last used: {last_used_time_str})")

            if least_recent_file in history:
                history.pop(least_recent_file)
                self._save_history(history)
            return True
        except Exception as e:
            print(f"Error deleting LoRA file {least_recent_file}: {e}")
            # If deletion fails, try to remove from history anyway if it was an OS error
            # but the file might be gone or inaccessible.
            if least_recent_file in history and not os.path.exists(lora_path_to_delete):
                history.pop(least_recent_file)
                self._save_history(history)
            return False

    @classmethod
    def INPUT_TYPES(s):
        max_lora_num = 10  # Max number of LoRAs user can specify
        inputs = {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "toggle": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "num_loras": ("INT", {"default": 1, "min": 1, "max": max_lora_num, "step": 1}),
            },
            "optional": {},  # Dynamic inputs will be added here
        }

        for i in range(1, max_lora_num + 1):
            inputs["optional"][f"lora_{i}_url"] = ("STRING", {
                "default": "",
                "multiline": True,
                "dynamicPrompts": False,  # Assuming URLs are static
                # Conditional visibility based on num_loras
                "forceInput": False,  # Make it an optional input socket if not specified in UI
            })
            inputs["optional"][f"lora_{i}_strength_model"] = ("FLOAT", {
                "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
            })
            inputs["optional"][f"lora_{i}_strength_clip"] = ("FLOAT", {
                "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
            })
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "load_and_apply_loras"

    CATEGORY = "EasyUse/Loaders/Video"  # Or your preferred category

    def download_lora(self, url):
        """Download a LoRA file from URL or copy from local path with disk space management"""
        filename = ""
        try:
            if url.startswith('http'):
                # Attempt to derive filename from URL
                path_part = url.split('?')[0]
                filename = os.path.basename(path_part)
                if not filename or '.' not in filename:  # Basic check for valid filename
                    # Use hash if no good filename can be derived
                    filename_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
                    filename = f"{filename_hash}.safetensors"  # Assume safetensors, could be .pt too
            elif os.path.exists(url):  # Local path
                filename = os.path.basename(url)
            else:  # Invalid URL or local path
                print(f"Invalid LoRA URL or local path: {url}")
                return None

            # Normalize filename to avoid issues, e.g. replace spaces
            filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in filename)
            if not filename:  # Fallback if normalization results in empty string
                filename_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
                filename = f"{filename_hash}.safetensors"

            full_path = os.path.join(self.lora_folder, filename)

            if os.path.exists(full_path):
                print(f"LoRA file {filename} already exists, skipping download.")
                self._update_lora_usage(filename)
                return filename

            # Check disk space and determine which threshold to use
            free_space_bytes, is_reliable = self._check_disk_space()
            free_space_mb = free_space_bytes / (1024 * 1024)

            # Set minimum free space based on disk reliability
            # Unreliable (network volume): keep 10 GB free
            # Reliable (normal disk): keep 5 GB free
            MIN_FREE_SPACE_MB = 5120 if is_reliable else 10240  # 5GB or 10GB in MB

            # Estimate required space (very rough, actual size unknown until download)
            # Let's assume an average LoRA size for initial check or just use a fixed buffer
            estimated_lora_size_mb = 150  # A common LoRA size in MB

            if free_space_mb < MIN_FREE_SPACE_MB + estimated_lora_size_mb:
                print(
                    f"Low disk space: {free_space_mb:.2f}MB available. Need ~{estimated_lora_size_mb}MB + {MIN_FREE_SPACE_MB}MB buffer.")
                while free_space_mb < MIN_FREE_SPACE_MB + estimated_lora_size_mb:
                    deleted = self._delete_least_recently_used_lora()
                    if not deleted:
                        print(f"Could not free up enough space for new LoRA from {url}. Download aborted.")
                        return None
                    free_space_bytes, is_reliable = self._check_disk_space()
                    free_space_mb = free_space_bytes / (1024 * 1024)
                    if free_space_mb >= MIN_FREE_SPACE_MB + estimated_lora_size_mb:
                        print(f"Freed up space. Now {free_space_mb:.2f}MB available.")
                        break
                if free_space_mb < MIN_FREE_SPACE_MB + estimated_lora_size_mb:  # Final check
                    print(f"Still not enough space after cleanup for {url}. Download aborted.")
                    return None

            if url.startswith('http'):
                print(f"Downloading LoRA from {url} to {filename}")
                response = requests.get(url, stream=True)
                response.raise_for_status()

                downloaded_size = 0
                with open(full_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded_size += len(chunk)

                downloaded_size_mb = downloaded_size / (1024 * 1024)
                print(f"Downloaded {filename} ({downloaded_size_mb:.2f} MB) to {full_path}")
                self._update_lora_usage(filename)
                return filename

            elif os.path.exists(url):  # Local path
                print(f"Copying LoRA from {url} to {full_path}")
                shutil.copy2(url, full_path)
                self._update_lora_usage(filename)
                print(f"Copied {filename} to {full_path}")
                return filename

            return None  # Should not reach here if logic is correct

        except requests.exceptions.RequestException as e:
            print(f"Error downloading LoRA from {url}: {e}")
            if filename and os.path.exists(os.path.join(self.lora_folder, filename)):  # Cleanup partial download
                try:
                    os.remove(os.path.join(self.lora_folder, filename))
                    print(f"Removed partially downloaded file: {filename}")
                except Exception as rm_e:
                    print(f"Error removing partially downloaded file {filename}: {rm_e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred with LoRA {url} (filename: {filename}): {e}")
            return None

    def load_and_apply_loras(self, model, clip, toggle, num_loras, **kwargs):
        if not toggle:
            return (model, clip)

        current_model = model
        current_clip = clip

        for i in range(1, int(num_loras) + 1):
            lora_url = kwargs.get(f"lora_{i}_url", "").strip()
            strength_model = float(kwargs.get(f"lora_{i}_strength_model", 1.0))
            strength_clip = float(kwargs.get(f"lora_{i}_strength_clip", 1.0))

            if not lora_url:
                continue

            if strength_model == 0 and strength_clip == 0:
                print(f"Skipping LoRA from {lora_url} as both strengths are 0.")
                continue

            lora_filename = self.download_lora(lora_url)

            if lora_filename:
                print(
                    f"Applying LoRA: {lora_filename} with model_strength: {strength_model}, clip_strength: {strength_clip}")
                try:
                    # LoraLoader().load_lora expects just the filename, it will find it in the loras folder
                    current_model, current_clip = self.lora_loader.load_lora(
                        current_model, current_clip, lora_filename, strength_model, strength_clip
                    )
                    # Usage is already updated by download_lora, but good to confirm.
                    # self._update_lora_usage(lora_filename) # Already called in download_lora
                except Exception as e:
                    print(f"Failed to apply LoRA {lora_filename} from {lora_url}: {e}")
            else:
                print(f"Failed to download or locate LoRA from {lora_url}, skipping application.")

        # Refresh ComfyUI's internal list of loras
        folder_paths.get_filename_list("loras")
        # Also, if subdirectories were used for loras (they can be), refresh them.
        # However, this node downloads directly to the main lora_folder.
        # folder_paths.refresh_custom_paths("loras") # More thorough but might be overkill

        return (current_model, current_clip)


class LoadVideoLoraByUrlOrPathSelect:
    def __init__(self):
        self.lora_folder = folder_paths.get_folder_paths("loras")[0]
        self._ensure_history_file()

    def _ensure_history_file(self):
        history_path = self._get_history_path()
        if not os.path.exists(history_path):
            self._save_history({})

    def _get_history_path(self):
        return os.path.join(self.lora_folder, ".lora_usage_history.json")

    def _load_history(self):
        history_path = self._get_history_path()
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_history(self, history):
        history_path = self._get_history_path()
        with open(history_path, 'w') as f:
            json.dump(history, f)

    def _update_lora_usage(self, lora_name):
        history = self._load_history()
        history[lora_name] = time.time()
        self._save_history(history)

    def _get_actual_used_space(self):
        """Calculate actual used space in the LoRA folder using du command"""
        import subprocess

        # Use du -sb for bytes output (more reliable than -sh for parsing)
        result = subprocess.run(
            ['du', '-sb', self.lora_folder],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Output format: "12345678\t/path/to/folder"
        size_str = result.stdout.split()[0]
        total_size = int(size_str)
        print(f"du command result: {total_size / (1024 ** 3):.2f}GB for {self.lora_folder}")
        return total_size

    def _check_disk_space(self):
        """Check available disk space in the LoRA folder
        Returns: tuple (free_space_bytes, is_reliable)
        - free_space_bytes: available space in bytes
        - is_reliable: True if disk reading is trustworthy, False if unreliable
        """
        try:
            total, used, free = shutil.disk_usage(self.lora_folder)
            free_gb = free / (1024 ** 3)

            # If free space > 200 GB, it's unreliable (network volume issue)
            if free_gb > 200:
                print(f"Unreliable disk reading detected ({free_gb:.1f} GB free). Using calculated space.")
                # Use hardcoded max of 200 GB
                MAX_DISK_SPACE_GB = 200
                actual_used_bytes = self._get_actual_used_space()
                actual_used_gb = actual_used_bytes / (1024 ** 3)
                free_space = (MAX_DISK_SPACE_GB * 1024 ** 3) - actual_used_bytes
                print(f"Actual used space: {actual_used_gb:.2f} GB, Calculated free: {free_space / (1024 ** 3):.2f} GB")
                return (free_space, False)  # False = unreliable system reading
            else:
                # Trust the system reading
                print(f"Reliable disk reading: {free_gb:.2f} GB free")
                return (free, True)  # True = reliable system reading
        except Exception as e:
            print(f"Error checking disk space: {e}")
            return (0, True)

    def _delete_least_recently_used_lora(self):
        history = self._load_history()
        lora_files = [f for f in os.listdir(self.lora_folder)
                      if os.path.isfile(os.path.join(self.lora_folder, f))
                      and not f.startswith('.')
                      and f != os.path.basename(self._get_history_path())]

        valid_history = {k: v for k, v in history.items() if k in lora_files}

        if not valid_history or not lora_files:
            print("No LoRA files available for deletion based on history or existing files.")
            return False

        least_recent_lora_entry = min(valid_history.items(), key=lambda x: x[1])
        least_recent_file = least_recent_lora_entry[0]

        lora_path_to_delete = os.path.join(self.lora_folder, least_recent_file)
        if not os.path.exists(lora_path_to_delete):
            print(f"Skipping deletion of {least_recent_file}, file not found (already deleted or moved).")
            if least_recent_file in history:
                history.pop(least_recent_file)
                self._save_history(history)
            return False

        last_used_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(least_recent_lora_entry[1]))

        try:
            file_size_mb = os.path.getsize(lora_path_to_delete) / (1024 * 1024)
            os.remove(lora_path_to_delete)
            print(f"Deleted least recently used LoRA: {least_recent_file} "
                  f"({file_size_mb:.2f} MB, last used: {last_used_time_str})")

            if least_recent_file in history:
                history.pop(least_recent_file)
                self._save_history(history)
            return True
        except Exception as e:
            print(f"Error deleting LoRA file {least_recent_file}: {e}")
            if least_recent_file in history and not os.path.exists(lora_path_to_delete):
                history.pop(least_recent_file)
                self._save_history(history)
            return False

    @classmethod
    def INPUT_TYPES(s):
        max_lora_num = 10
        inputs = {
            "required": {
                "toggle": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "num_loras": ("INT", {"default": 1, "min": 1, "max": max_lora_num, "step": 1}),
            },
            "optional": {
                "prev_lora": ("WANVIDLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
            },
        }

        for i in range(1, max_lora_num + 1):
            inputs["optional"][f"lora_{i}_url"] = ("STRING", {
                "default": "",
                "multiline": True,
                "dynamicPrompts": False,
                "forceInput": False,
            })
            inputs["optional"][f"lora_{i}_strength"] = ("FLOAT", {
                "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
            })
        return inputs

    RETURN_TYPES = ("WANVIDLORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "load_and_select_loras"
    CATEGORY = "EasyUse/Loaders/Video"

    def download_lora(self, url):
        filename = ""
        try:
            if url.startswith('http'):
                path_part = url.split('?')[0]
                filename = os.path.basename(path_part)
                if not filename or '.' not in filename:
                    filename_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
                    filename = f"{filename_hash}.safetensors"
            elif os.path.exists(url):
                filename = os.path.basename(url)
            else:
                print(f"Invalid LoRA URL or local path: {url}")
                return None

            filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in filename)
            if not filename:
                filename_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
                filename = f"{filename_hash}.safetensors"

            full_path = os.path.join(self.lora_folder, filename)

            if os.path.exists(full_path):
                print(f"LoRA file {filename} already exists, skipping download.")
                self._update_lora_usage(filename)
                return filename

            # Check disk space and determine which threshold to use
            free_space_bytes, is_reliable = self._check_disk_space()
            free_space_mb = free_space_bytes / (1024 * 1024)

            # Set minimum free space based on disk reliability
            # Unreliable (network volume): keep 10 GB free
            # Reliable (normal disk): keep 5 GB free
            MIN_FREE_SPACE_MB = 5120 if is_reliable else 10240  # 5GB or 10GB in MB
            estimated_lora_size_mb = 150

            if free_space_mb < MIN_FREE_SPACE_MB + estimated_lora_size_mb:
                print(f"Low disk space: {free_space_mb:.2f}MB available. Need ~{estimated_lora_size_mb}MB + {MIN_FREE_SPACE_MB}MB buffer.")
                while free_space_mb < MIN_FREE_SPACE_MB + estimated_lora_size_mb:
                    deleted = self._delete_least_recently_used_lora()
                    if not deleted:
                        print(f"Could not free up enough space for new LoRA from {url}. Download aborted.")
                        return None
                    free_space_bytes, is_reliable = self._check_disk_space()
                    free_space_mb = free_space_bytes / (1024 * 1024)
                    if free_space_mb >= MIN_FREE_SPACE_MB + estimated_lora_size_mb:
                        print(f"Freed up space. Now {free_space_mb:.2f}MB available.")
                        break
                if free_space_mb < MIN_FREE_SPACE_MB + estimated_lora_size_mb:
                    print(f"Still not enough space after cleanup for {url}. Download aborted.")
                    return None

            if url.startswith('http'):
                print(f"Downloading LoRA from {url} to {filename}")
                response = requests.get(url, stream=True)
                response.raise_for_status()

                downloaded_size = 0
                with open(full_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded_size += len(chunk)

                downloaded_size_mb = downloaded_size / (1024 * 1024)
                print(f"Downloaded {filename} ({downloaded_size_mb:.2f} MB) to {full_path}")
                self._update_lora_usage(filename)
                return filename

            elif os.path.exists(url):
                print(f"Copying LoRA from {url} to {full_path}")
                shutil.copy2(url, full_path)
                self._update_lora_usage(filename)
                print(f"Copied {filename} to {full_path}")
                return filename

            return None

        except requests.exceptions.RequestException as e:
            print(f"Error downloading LoRA from {url}: {e}")
            if filename and os.path.exists(os.path.join(self.lora_folder, filename)):
                try:
                    os.remove(os.path.join(self.lora_folder, filename))
                    print(f"Removed partially downloaded file: {filename}")
                except Exception as rm_e:
                    print(f"Error removing partially downloaded file {filename}: {rm_e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred with LoRA {url} (filename: {filename}): {e}")
            return None

    def load_and_select_loras(self, toggle, num_loras, prev_lora=None, **kwargs):
        if not toggle:
            return (prev_lora or [],)

        loras_list = list(prev_lora) if prev_lora else []

        for i in range(1, int(num_loras) + 1):
            lora_url = kwargs.get(f"lora_{i}_url", "").strip()
            strength = float(kwargs.get(f"lora_{i}_strength", 1.0))

            if not lora_url or strength == 0.0:
                continue

            lora_filename = self.download_lora(lora_url)

            if lora_filename:
                print(f"Adding LoRA: {lora_filename} with strength: {strength}")
                loras_list.append({
                    "path": os.path.join(self.lora_folder, lora_filename),
                    "strength": round(strength, 4),
                    "name": os.path.splitext(lora_filename)[0],
                    "blocks": {},
                    "layer_filter": "",
                    "low_mem_load": False,
                    "merge_loras": False,
                })
            else:
                print(f"Failed to download or locate LoRA from {lora_url}, skipping.")

        folder_paths.get_filename_list("loras")
        return (loras_list,)