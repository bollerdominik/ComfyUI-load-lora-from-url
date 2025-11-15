import hashlib
import json
import os
import shutil
import time

import folder_paths
import requests
from nodes import LoraLoader  # Import LoraLoader for applying LoRAs


# This ComfyUI node allows you to load Video LoRA files from URLs or local paths
# and apply them to a model and clip.
# It downloads the files if they are not already present in the specified folder.
# It also manages disk space by tracking usage and removing least recently used files when space is low.
class LoadVideoLoraByUrlOrPath:
    def __init__(self):
        # Initialize lora folder path
        self.lora_folder = folder_paths.get_folder_paths("loras")[0]
        self.history_file = os.path.join(self.lora_folder, "history.json")
        self.min_free_space_gb = 2  # Minimum free space in GB

        # Protected LoRAs that should never be deleted
        self.protected_keywords = ["lightning"]  # Case insensitive matching

        # Network volume settings
        self.network_volume_path = "/workspace/network-volume"  # Path to check for network volume
        self.network_volume_free_space_threshold_gb = 300  # If free space > this, consider unreliable
        self.network_volume_max_size_gb = 190  # Maximum total size for network volumes

        # Initialize LoraLoader for applying LoRAs
        self.lora_loader = LoraLoader()

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

    CATEGORY = "EasyUse/Loaders/Video"

    def download_lora(self, url):
        """Download a LoRA file from URL or copy from local path"""
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
                # Validate existing file before using it. delete files with 0 bytes
                if self._validate_and_cleanup_file(full_path, filename):
                    print(f"Video LoRA file {filename} already exists, skipping download")
                    # Update usage history for existing file
                    self._update_lora_usage(filename)
                    return filename
                else:
                    print(f"Existing Video LoRA file {filename} was invalid and removed, re-downloading...")

            # Download the file
            if url.startswith('http'):
                print(f"Downloading Video LoRA from {url}")
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
                print(f"Downloaded Video LoRA to {full_path}")

                # Validate the downloaded file
                if not self._validate_and_cleanup_file(full_path, filename):
                    print(f"Downloaded Video LoRA file {filename} was invalid and removed")
                    return None

                # Update usage history for the new file
                self._update_lora_usage(filename)
                return filename
            else:
                # Copy local file if it's a valid path
                if os.path.exists(url):
                    shutil.copy2(url, full_path)
                    print(f"Copied Video LoRA from {url} to {full_path}")

                    # Validate the copied file
                    if not self._validate_and_cleanup_file(full_path, filename):
                        print(f"Copied Video LoRA file {filename} was invalid and removed")
                        return None

                    # Update usage history for the new file
                    self._update_lora_usage(filename)
                    return filename
                else:
                    print(f"Local Video LoRA file {url} does not exist")
                    return None

        except Exception as e:
            print(f"Error downloading Video LoRA: {e}")
            return None

    def load_and_apply_loras(self, model, clip, toggle, num_loras, **kwargs):
        """Load and apply Video LoRA files to model and clip with usage tracking"""
        if toggle in [False, None, "False"]:
            return (model, clip)

        # Process each LoRA
        for i in range(1, num_loras + 1):
            lora_url = kwargs.get(f"lora_{i}_url", "")

            if not lora_url:
                continue

            # Check disk space and remove old loras if needed
            self._manage_disk_space()

            # Download/copy the LoRA file with disk space management
            lora_name = self.download_lora(lora_url)

            if not lora_name:
                continue

            # Get strength values
            model_strength = float(kwargs.get(f"lora_{i}_strength_model", 1.0))
            clip_strength = float(kwargs.get(f"lora_{i}_strength_clip", 1.0))

            # Apply the LoRA using LoraLoader
            try:
                model, clip = self.lora_loader.load_lora(model, clip, lora_name, model_strength, clip_strength)
                print(f"Applied Video LoRA {lora_name} with strengths - model: {model_strength}, clip: {clip_strength}")
            except Exception as e:
                print(f"Error applying Video LoRA {lora_name}: {e}")
                continue

        # Refresh the lora list to include newly downloaded files
        folder_paths.get_filename_list("loras")

        return (model, clip)

    def _load_history(self):
        """Load LoRA usage history from JSON file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                    print(f"Loaded Video LoRA history: {len(history)} entries")
                    print(f"History content: {json.dumps(history, indent=2)}")
                    return history
            except Exception as e:
                print(f"Error loading history file: {e}")
                return {}
        else:
            print("No history file found, starting fresh")
            return {}

    def _save_history(self, history):
        """Save LoRA usage history to JSON file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Error saving history file: {e}")

    def _update_lora_usage(self, filename):
        """Update the last usage timestamp for a LoRA file"""
        history = self._load_history()
        history[filename] = time.time()
        self._save_history(history)
        print(f"Updated usage timestamp for {filename}")

    def _get_disk_space(self):
        """Get available disk space in bytes for the lora folder"""
        try:
            stat = shutil.disk_usage(self.lora_folder)
            return stat.free
        except Exception as e:
            print(f"Error getting disk space: {e}")
            return None

    def _manage_disk_space(self):
        """Check disk space and remove old LoRAs if below threshold"""
        free_space = self._get_disk_space()
        if free_space is None:
            return

        free_space_gb = free_space / (1024 * 1024 * 1024)
        print(f"Free space: {free_space_gb:.2f} GB")

        # Check if we're on a network volume with unreliable free space reporting
        if free_space_gb > self.network_volume_free_space_threshold_gb:
            print(f"Free space > {self.network_volume_free_space_threshold_gb} GB - may be unreliable network volume")
            print("Switching to total folder size check...")
            self._manage_network_volume_space()
        else:
            # Normal disk space management
            min_free_bytes = self.min_free_space_gb * 1024 * 1024 * 1024
            if free_space < min_free_bytes:
                print(f"Low disk space detected. Free space: {free_space_gb:.2f} GB")
                print(f"Attempting to free up space to reach {self.min_free_space_gb} GB...")
                self._remove_old_loras(min_free_bytes - free_space)

    def _is_protected(self, filename):
        """Check if a file is protected from deletion"""
        filename_lower = filename.lower()
        for keyword in self.protected_keywords:
            if keyword.lower() in filename_lower:
                return True
        return False

    def _get_folder_size(self, folder_path=None):
        """Get total size of a folder in bytes (recursively)"""
        if folder_path is None:
            folder_path = self.lora_folder

        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except Exception as e:
                        print(f"Error getting size of {filepath}: {e}")
        except Exception as e:
            print(f"Error calculating folder size for {folder_path}: {e}")
        return total_size

    def _manage_network_volume_space(self):
        """Manage space for network volumes by checking total network volume size"""
        # Check if network volume path exists
        if not os.path.exists(self.network_volume_path):
            print(f"Network volume path {self.network_volume_path} does not exist")
            print("Falling back to normal disk space management")
            return

        # Get total size of the entire network volume
        total_size = self._get_folder_size(self.network_volume_path)
        total_size_gb = total_size / (1024 * 1024 * 1024)
        print(f"Total network volume size ({self.network_volume_path}): {total_size_gb:.2f} GB")

        if total_size_gb > self.network_volume_max_size_gb:
            bytes_to_free = total_size - (self.network_volume_max_size_gb * 1024 * 1024 * 1024)
            print(f"Network volume size exceeds {self.network_volume_max_size_gb} GB threshold")
            print(f"Attempting to free {bytes_to_free / (1024 * 1024 * 1024):.2f} GB from LoRA folder...")
            self._remove_old_loras(bytes_to_free)
        else:
            print(f"Network volume size within limits ({self.network_volume_max_size_gb} GB)")

    def _remove_old_loras(self, bytes_needed):
        """Remove least recently used LoRAs until enough space is freed (skips protected files)"""
        history = self._load_history()

        # Get all LoRA files in the folder with their timestamps
        lora_files = []
        protected_files = []

        for filename in os.listdir(self.lora_folder):
            filepath = os.path.join(self.lora_folder, filename)
            # Skip directories and the history file
            if os.path.isfile(filepath) and filename != "video_lora_history.json":
                # Check if file is protected
                if self._is_protected(filename):
                    protected_files.append(filename)
                    print(f"Skipping protected Video LoRA: {filename}")
                    continue

                # Use history timestamp if available, otherwise use file modification time
                timestamp = history.get(filename, os.path.getmtime(filepath))
                file_size = os.path.getsize(filepath)
                lora_files.append((filename, timestamp, file_size, filepath))

        # Sort by timestamp (oldest first)
        lora_files.sort(key=lambda x: x[1])

        bytes_freed = 0
        removed_files = []

        for filename, timestamp, file_size, filepath in lora_files:
            if bytes_freed >= bytes_needed:
                break

            try:
                os.remove(filepath)
                bytes_freed += file_size
                removed_files.append(filename)
                print(f"Removed Video LoRA: {filename} (freed {file_size / (1024 * 1024):.2f} MB)")

                # Remove from history
                if filename in history:
                    del history[filename]

            except Exception as e:
                print(f"Error removing file {filename}: {e}")

        # Save updated history
        if removed_files:
            self._save_history(history)
            print(f"Total space freed: {bytes_freed / (1024 * 1024):.2f} MB")
            print(f"Removed {len(removed_files)} Video LoRA file(s)")

        # Warn if we couldn't free enough space
        if bytes_freed < bytes_needed:
            shortage_gb = (bytes_needed - bytes_freed) / (1024 * 1024 * 1024)
            print(f"WARNING: Could not free enough space. Still need {shortage_gb:.2f} GB more.")
            if protected_files:
                print(f"Note: {len(protected_files)} protected file(s) were skipped from deletion.")

    def _validate_and_cleanup_file(self, full_path, filename):
        """Validate a LoRA file and remove if invalid (e.g., 0 bytes)"""
        try:
            file_size = os.path.getsize(full_path)
            if file_size == 0:
                print(f"File {filename} is 0 bytes, removing...")
                os.remove(full_path)
                return False
            return True
        except Exception as e:
            print(f"Error validating file {filename}: {e}")
            return False


# Node for loading Video LoRAs into WANVIDLORA format (for Wan Video LoRA workflow)
class LoadVideoLoraByUrlOrPathSelect:
    def __init__(self):
        # Initialize lora folder path
        self.lora_folder = folder_paths.get_folder_paths("loras")[0]
        self.history_file = os.path.join(self.lora_folder, "history.json")
        self.min_free_space_gb = 2  # Minimum free space in GB

        # Protected LoRAs that should never be deleted
        self.protected_keywords = ["lightning"]  # Case insensitive matching

        # Network volume settings
        self.network_volume_path = "/workspace/network-volume"  # Path to check for network volume
        self.network_volume_free_space_threshold_gb = 300  # If free space > this, consider unreliable
        self.network_volume_max_size_gb = 190  # Maximum total size for network volumes

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
        """Download a LoRA file from URL or copy from local path"""
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
                # Validate existing file before using it. delete files with 0 bytes
                if self._validate_and_cleanup_file(full_path, filename):
                    print(f"Video LoRA file {filename} already exists, skipping download")
                    # Update usage history for existing file
                    self._update_lora_usage(filename)
                    return filename
                else:
                    print(f"Existing Video LoRA file {filename} was invalid and removed, re-downloading...")

            # Download the file
            if url.startswith('http'):
                print(f"Downloading Video LoRA from {url}")
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
                print(f"Downloaded Video LoRA to {full_path}")

                # Validate the downloaded file
                if not self._validate_and_cleanup_file(full_path, filename):
                    print(f"Downloaded Video LoRA file {filename} was invalid and removed")
                    return None

                # Update usage history for the new file
                self._update_lora_usage(filename)
                return filename
            else:
                # Copy local file if it's a valid path
                if os.path.exists(url):
                    shutil.copy2(url, full_path)
                    print(f"Copied Video LoRA from {url} to {full_path}")

                    # Validate the copied file
                    if not self._validate_and_cleanup_file(full_path, filename):
                        print(f"Copied Video LoRA file {filename} was invalid and removed")
                        return None

                    # Update usage history for the new file
                    self._update_lora_usage(filename)
                    return filename
                else:
                    print(f"Local Video LoRA file {url} does not exist")
                    return None

        except Exception as e:
            print(f"Error downloading Video LoRA: {e}")
            return None

    def load_and_select_loras(self, toggle, num_loras, prev_lora=None, **kwargs):
        """Load Video LoRA files and return as WANVIDLORA format with usage tracking"""
        if not toggle:
            return (prev_lora or [],)

        # Start with previous loras if provided
        loras_list = list(prev_lora) if prev_lora else []

        # Process each LoRA
        for i in range(1, int(num_loras) + 1):
            lora_url = kwargs.get(f"lora_{i}_url", "").strip()
            strength = float(kwargs.get(f"lora_{i}_strength", 1.0))

            if not lora_url or strength == 0.0:
                continue

            # Check disk space and remove old loras if needed
            self._manage_disk_space()

            # Download/copy the LoRA file with disk space management
            lora_filename = self.download_lora(lora_url)

            if not lora_filename:
                print(f"Failed to download or locate LoRA from {lora_url}, skipping.")
                continue

            # Add to lora list as dictionary (WANVIDLORA format)
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

        # Refresh the lora list to include newly downloaded files
        folder_paths.get_filename_list("loras")

        return (loras_list,)

    def _load_history(self):
        """Load LoRA usage history from JSON file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                    print(f"Loaded Video LoRA history: {len(history)} entries")
                    print(f"History content: {json.dumps(history, indent=2)}")
                    return history
            except Exception as e:
                print(f"Error loading history file: {e}")
                return {}
        else:
            print("No history file found, starting fresh")
            return {}

    def _save_history(self, history):
        """Save LoRA usage history to JSON file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Error saving history file: {e}")

    def _update_lora_usage(self, filename):
        """Update the last usage timestamp for a LoRA file"""
        history = self._load_history()
        history[filename] = time.time()
        self._save_history(history)
        print(f"Updated usage timestamp for {filename}")

    def _get_disk_space(self):
        """Get available disk space in bytes for the lora folder"""
        try:
            stat = shutil.disk_usage(self.lora_folder)
            return stat.free
        except Exception as e:
            print(f"Error getting disk space: {e}")
            return None

    def _manage_disk_space(self):
        """Check disk space and remove old LoRAs if below threshold"""
        free_space = self._get_disk_space()
        if free_space is None:
            return

        free_space_gb = free_space / (1024 * 1024 * 1024)
        print(f"Free space: {free_space_gb:.2f} GB")

        # Check if we're on a network volume with unreliable free space reporting
        if free_space_gb > self.network_volume_free_space_threshold_gb:
            print(f"Free space > {self.network_volume_free_space_threshold_gb} GB - may be unreliable network volume")
            print("Switching to total folder size check...")
            self._manage_network_volume_space()
        else:
            # Normal disk space management
            min_free_bytes = self.min_free_space_gb * 1024 * 1024 * 1024
            if free_space < min_free_bytes:
                print(f"Low disk space detected. Free space: {free_space_gb:.2f} GB")
                print(f"Attempting to free up space to reach {self.min_free_space_gb} GB...")
                self._remove_old_loras(min_free_bytes - free_space)

    def _is_protected(self, filename):
        """Check if a file is protected from deletion"""
        filename_lower = filename.lower()
        for keyword in self.protected_keywords:
            if keyword.lower() in filename_lower:
                return True
        return False

    def _get_folder_size(self, folder_path=None):
        """Get total size of a folder in bytes (recursively)"""
        if folder_path is None:
            folder_path = self.lora_folder

        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except Exception as e:
                        print(f"Error getting size of {filepath}: {e}")
        except Exception as e:
            print(f"Error calculating folder size for {folder_path}: {e}")
        return total_size

    def _manage_network_volume_space(self):
        """Manage space for network volumes by checking total network volume size"""
        # Check if network volume path exists
        if not os.path.exists(self.network_volume_path):
            print(f"Network volume path {self.network_volume_path} does not exist")
            print("Falling back to normal disk space management")
            return

        # Get total size of the entire network volume
        total_size = self._get_folder_size(self.network_volume_path)
        total_size_gb = total_size / (1024 * 1024 * 1024)
        print(f"Total network volume size ({self.network_volume_path}): {total_size_gb:.2f} GB")

        if total_size_gb > self.network_volume_max_size_gb:
            bytes_to_free = total_size - (self.network_volume_max_size_gb * 1024 * 1024 * 1024)
            print(f"Network volume size exceeds {self.network_volume_max_size_gb} GB threshold")
            print(f"Attempting to free {bytes_to_free / (1024 * 1024 * 1024):.2f} GB from LoRA folder...")
            self._remove_old_loras(bytes_to_free)
        else:
            print(f"Network volume size within limits ({self.network_volume_max_size_gb} GB)")

    def _remove_old_loras(self, bytes_needed):
        """Remove least recently used LoRAs until enough space is freed (skips protected files)"""
        history = self._load_history()

        # Get all LoRA files in the folder with their timestamps
        lora_files = []
        protected_files = []

        for filename in os.listdir(self.lora_folder):
            filepath = os.path.join(self.lora_folder, filename)
            # Skip directories and the history file
            if os.path.isfile(filepath) and filename != "video_lora_history.json":
                # Check if file is protected
                if self._is_protected(filename):
                    protected_files.append(filename)
                    print(f"Skipping protected Video LoRA: {filename}")
                    continue

                # Use history timestamp if available, otherwise use file modification time
                timestamp = history.get(filename, os.path.getmtime(filepath))
                file_size = os.path.getsize(filepath)
                lora_files.append((filename, timestamp, file_size, filepath))

        # Sort by timestamp (oldest first)
        lora_files.sort(key=lambda x: x[1])

        bytes_freed = 0
        removed_files = []

        for filename, timestamp, file_size, filepath in lora_files:
            if bytes_freed >= bytes_needed:
                break

            try:
                os.remove(filepath)
                bytes_freed += file_size
                removed_files.append(filename)
                print(f"Removed Video LoRA: {filename} (freed {file_size / (1024 * 1024):.2f} MB)")

                # Remove from history
                if filename in history:
                    del history[filename]

            except Exception as e:
                print(f"Error removing file {filename}: {e}")

        # Save updated history
        if removed_files:
            self._save_history(history)
            print(f"Total space freed: {bytes_freed / (1024 * 1024):.2f} MB")
            print(f"Removed {len(removed_files)} Video LoRA file(s)")

        # Warn if we couldn't free enough space
        if bytes_freed < bytes_needed:
            shortage_gb = (bytes_needed - bytes_freed) / (1024 * 1024 * 1024)
            print(f"WARNING: Could not free enough space. Still need {shortage_gb:.2f} GB more.")
            if protected_files:
                print(f"Note: {len(protected_files)} protected file(s) were skipped from deletion.")

    def _validate_and_cleanup_file(self, full_path, filename):
        """Validate a LoRA file and remove if invalid (e.g., 0 bytes)"""
        try:
            file_size = os.path.getsize(full_path)
            if file_size == 0:
                print(f"File {filename} is 0 bytes, removing...")
                os.remove(full_path)
                return False
            return True
        except Exception as e:
            print(f"Error validating file {filename}: {e}")
            return False