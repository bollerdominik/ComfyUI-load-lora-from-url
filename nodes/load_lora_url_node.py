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

        # Volume size limit in bytes (100GB = 100 * 1024^3)
        self.VOLUME_SIZE_LIMIT = 100 * 1024 * 1024 * 1024
        # Threshold before cleanup (93GB = 93 * 1024^3)
        self.VOLUME_CLEANUP_THRESHOLD = 93 * 1024 * 1024 * 1024

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

    def _get_volume_root(self):
        """Detect the mounted volume root path"""
        # Try to find the network-volume path by going up from lora_folder
        current_path = os.path.abspath(self.lora_folder)

        # Look for common mounted volume patterns
        volume_indicators = ['network-volume', 'workspace']

        while current_path != '/':
            folder_name = os.path.basename(current_path)
            if any(indicator in folder_name for indicator in volume_indicators):
                return current_path
            current_path = os.path.dirname(current_path)

        # If no specific volume found, use the lora folder itself
        return self.lora_folder

    def _calculate_folder_size(self, folder_path):
        """Calculate total size of a folder and all its contents"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(file_path)
                    except (OSError, IOError):
                        # Skip files that can't be accessed
                        continue
            return total_size
        except Exception as e:
            print(f"Error calculating folder size for {folder_path}: {e}")
            return 0

    def _check_disk_space(self):
        """Check available disk space in the LoRA folder"""
        try:
            total, used, free = shutil.disk_usage(self.lora_folder)
            return free
        except Exception as e:
            print(f"Error checking disk space: {e}")
            return 0

    def _check_volume_size(self):
        """Check the actual volume usage for mounted filesystems"""
        try:
            volume_root = self._get_volume_root()
            current_size = self._calculate_folder_size(volume_root)

            print(f"Volume root: {volume_root}")
            print(f"Current volume usage: {current_size / (1024 ** 3):.2f}GB")
            print(f"Volume limit: {self.VOLUME_SIZE_LIMIT / (1024 ** 3):.2f}GB")
            print(f"Cleanup threshold: {self.VOLUME_CLEANUP_THRESHOLD / (1024 ** 3):.2f}GB")

            return current_size, volume_root
        except Exception as e:
            print(f"Error checking volume size: {e}")
            return 0, self.lora_folder

    def _delete_least_recently_used_lora(self):
        """Delete the least recently used LoRA file to free up space"""
        print("Starting LoRA cleanup process...")

        history = self._load_history()
        print(f"Loaded history with {len(history)} entries")

        # Get all existing LoRA files (excluding hidden files and the history file)
        lora_files = [f for f in os.listdir(self.lora_folder)
                      if os.path.isfile(os.path.join(self.lora_folder, f))
                      and not f.startswith('.')]

        print(f"Found {len(lora_files)} LoRA files in folder")

        # If no files at all, nothing to delete
        if not lora_files:
            print("No LoRA files available for deletion")
            return False

        # For files not in history, add them with their file modification time
        current_time = time.time()
        files_added_to_history = 0

        for lora_file in lora_files:
            if lora_file not in history:
                try:
                    file_path = os.path.join(self.lora_folder, lora_file)
                    # Use file modification time
                    mod_time = os.path.getmtime(file_path)
                    history[lora_file] = mod_time
                    files_added_to_history += 1
                    # Only print for first few files to avoid log spam
                    if files_added_to_history <= 5:
                        print(f"Added {lora_file} to history with modification time")
                    elif files_added_to_history == 6:
                        print("... (continuing to add more files to history)")
                except Exception as e:
                    # If we can't get mod time, use a very old timestamp to prioritize for deletion
                    history[lora_file] = current_time - (365 * 24 * 3600)  # 1 year ago
                    files_added_to_history += 1
                    print(f"Added {lora_file} to history with fallback timestamp: {e}")

        # Save the updated history if files were added
        if files_added_to_history > 0:
            self._save_history(history)
            print(f"Added {files_added_to_history} files to history and saved")

        # Now ALL existing files should be in history
        # Filter history to only include existing files (in case history has stale entries)
        valid_history = {k: v for k, v in history.items() if k in lora_files}

        print(f"Valid history entries: {len(valid_history)}")

        # Now we should have valid history for all existing files
        if not valid_history:
            print("ERROR: No valid history entries found after update")
            return False

        # Find least recently used LoRA
        least_recent_lora = min(valid_history.items(), key=lambda x: x[1])
        least_recent_file = least_recent_lora[0]
        last_used_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                       time.localtime(least_recent_lora[1]))

        print(f"Selected for deletion: {least_recent_file} (last used: {last_used_time})")

        # Delete the file
        try:
            lora_path = os.path.join(self.lora_folder, least_recent_file)

            # Check if file exists before trying to delete
            if not os.path.exists(lora_path):
                print(f"File {least_recent_file} no longer exists, removing from history")
                history.pop(least_recent_file, None)
                self._save_history(history)
                return False

            file_size = os.path.getsize(lora_path) / (1024 * 1024)  # Size in MB
            print(f"Deleting file: {lora_path} ({file_size:.2f} MB)")

            os.remove(lora_path)
            print(f"Successfully deleted LoRA: {least_recent_file} "
                  f"({file_size:.2f} MB, last used: {last_used_time})")

            # Update history by removing the deleted file
            history.pop(least_recent_file, None)
            self._save_history(history)

            # Verify file was actually deleted
            if os.path.exists(lora_path):
                print(f"ERROR: File {least_recent_file} still exists after deletion attempt!")
                return False
            else:
                print(f"Confirmed: File {least_recent_file} has been deleted")
                return True

        except Exception as e:
            print(f"ERROR deleting LoRA file {least_recent_file}: {e}")
            return False

    def _validate_and_cleanup_file(self, file_path, filename):
        """Check if file is valid (non-zero size) and delete if invalid"""
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                print(f"Downloaded file {filename} is 0 bytes, deleting...")
                os.remove(file_path)
                return False
            return True
        except Exception as e:
            print(f"Error validating file {filename}: {e}")
            # Try to delete the file if there was an error checking it
            try:
                os.remove(file_path)
            except:
                pass
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
                # Validate existing file before using it
                if self._validate_and_cleanup_file(full_path, filename):
                    print(f"LoRA file {filename} already exists, skipping download")
                    # Update usage history for existing file
                    self._update_lora_usage(filename)
                    return filename
                else:
                    print(f"Existing LoRA file {filename} was invalid and removed, re-downloading...")

            # Check available disk space before downloading (original method)
            MIN_FREE_SPACE = 2 * 1024 * 1024 * 1024  # 2GB in bytes
            free_space = self._check_disk_space()

            # Check volume size (new method for mounted filesystems)
            current_volume_size, volume_root = self._check_volume_size()

            # Determine if we need to free up space based on either check
            need_cleanup = False
            cleanup_reason = ""

            # Check traditional disk space
            if free_space < MIN_FREE_SPACE and free_space > 0:  # free_space > 0 means the check worked
                need_cleanup = True
                cleanup_reason = f"Low disk space: {free_space / (1024 ** 3):.2f}GB available. Need at least 2GB."

            # Check volume size (for mounted filesystems)
            if current_volume_size > self.VOLUME_CLEANUP_THRESHOLD:
                need_cleanup = True
                if cleanup_reason:
                    cleanup_reason += f" Also, "
                cleanup_reason += f"Volume size ({current_volume_size / (1024 ** 3):.2f}GB) exceeds threshold ({self.VOLUME_CLEANUP_THRESHOLD / (1024 ** 3):.2f}GB)."

            # If space is low based on either check, try to free up space
            if need_cleanup:
                print(cleanup_reason)
                print("Starting cleanup process...")

                # Try to delete least recently used LoRAs until enough space is freed
                cleanup_attempts = 0
                max_cleanup_attempts = 10  # Prevent infinite loops

                while cleanup_attempts < max_cleanup_attempts:
                    # Check current space before attempting cleanup
                    free_space = self._check_disk_space()
                    current_volume_size, _ = self._check_volume_size()

                    print(f"Cleanup attempt {cleanup_attempts + 1}:")
                    print(f"  Current free space: {free_space / (1024 ** 3):.2f}GB")
                    print(f"  Current volume size: {current_volume_size / (1024 ** 3):.2f}GB")

                    # Check if we've freed enough space
                    space_ok = (free_space >= MIN_FREE_SPACE or free_space == 0)  # free_space == 0 means check failed
                    volume_ok = current_volume_size <= self.VOLUME_CLEANUP_THRESHOLD

                    if space_ok and volume_ok:
                        print(
                            f"Cleanup complete! Disk: {free_space / (1024 ** 3):.2f}GB available, Volume: {current_volume_size / (1024 ** 3):.2f}GB used")
                        break

                    # Try to delete one file
                    deleted = self._delete_least_recently_used_lora()

                    if not deleted:
                        print(f"Could not delete any files on attempt {cleanup_attempts + 1}")
                        if cleanup_attempts == 0:
                            print("No files were deleted - this might be a permissions issue or all files are in use")
                        break

                    cleanup_attempts += 1
                    print(f"Cleanup attempt {cleanup_attempts} completed, checking space again...")

                # Final check after cleanup attempts
                if cleanup_attempts >= max_cleanup_attempts:
                    print(f"Reached maximum cleanup attempts ({max_cleanup_attempts}), stopping cleanup")

                final_free_space = self._check_disk_space()
                final_volume_size, _ = self._check_volume_size()
                final_space_ok = (final_free_space >= MIN_FREE_SPACE or final_free_space == 0)
                final_volume_ok = final_volume_size <= self.VOLUME_CLEANUP_THRESHOLD

                if not (final_space_ok and final_volume_ok):
                    print("Could not free up enough space for new LoRA download")
                    print(
                        f"Final state - Free space: {final_free_space / (1024 ** 3):.2f}GB, Volume size: {final_volume_size / (1024 ** 3):.2f}GB")
                    return None
                else:
                    print("Cleanup successful, proceeding with download")

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
                    import shutil
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