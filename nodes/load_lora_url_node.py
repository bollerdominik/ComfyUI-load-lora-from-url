import hashlib
import json
import os
import shutil
import time

import folder_paths
import requests


# This ComfyUI node allows you to load LoRA files from URLs or local paths.
# It downloads the files if they are not already present in the specified folder.
# It also allows you to stack multiple LoRA files together with different strengths.
# Additionally, it manages disk space by tracking usage and removing least recently used files when space is low.
class LoadLoraByUrlOrPath:
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
        """Download a LoRA file from URL or copy from local path """
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
                    print(f"LoRA file {filename} already exists, skipping download")
                    # Update usage history for existing file
                    self._update_lora_usage(filename)
                    return filename
                else:
                    print(f"Existing LoRA file {filename} was invalid and removed, re-downloading...")

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

                # Validate the downloaded file
                if not self._validate_and_cleanup_file(full_path, filename):
                    print(f"Downloaded LoRA file {filename} was invalid and removed")
                    return None

                # Update usage history for the new file
                self._update_lora_usage(filename)
                return filename
            else:
                # Copy local file if it's a valid path
                if os.path.exists(url):
                    shutil.copy2(url, full_path)
                    print(f"Copied LoRA from {url} to {full_path}")

                    # Validate the copied file
                    if not self._validate_and_cleanup_file(full_path, filename):
                        print(f"Copied LoRA file {filename} was invalid and removed")
                        return None

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

            # Check disk space and remove old loras if needed
            self._manage_disk_space()

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

    def _load_history(self):
        """Load LoRA usage history from JSON file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                    print(f"Loaded LoRA history: {len(history)} entries")
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
            if os.path.isfile(filepath) and filename != "history.json":
                # Check if file is protected
                if self._is_protected(filename):
                    protected_files.append(filename)
                    print(f"Skipping protected LoRA: {filename}")
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
                print(f"Removed LoRA: {filename} (freed {file_size / (1024 * 1024):.2f} MB)")

                # Remove from history
                if filename in history:
                    del history[filename]

            except Exception as e:
                print(f"Error removing file {filename}: {e}")

        # Save updated history
        if removed_files:
            self._save_history(history)
            print(f"Total space freed: {bytes_freed / (1024 * 1024):.2f} MB")
            print(f"Removed {len(removed_files)} LoRA file(s)")

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