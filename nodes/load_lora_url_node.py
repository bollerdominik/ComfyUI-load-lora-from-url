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
        self.VOLUME_SIZE_LIMIT = 145 * 1024 * 1024 * 1024
        # Threshold before cleanup (93GB = 93 * 1024^3)
        self.VOLUME_CLEANUP_THRESHOLD = 125 * 1024 * 1024 * 1024

    def _ensure_history_file(self):
        """Initialize the usage history file if it doesn't exist"""
        try:
            history_path = self._get_history_path()
            print(f"History file path: {history_path}")

            if not os.path.exists(history_path):
                print("History file does not exist, creating new one")
                self._save_history({})
            else:
                # Test if we can read the existing file
                try:
                    with open(history_path, 'r') as f:
                        test_data = json.load(f)
                    print(f"History file exists and is readable with {len(test_data)} entries")
                except Exception as e:
                    print(f"History file exists but is corrupted: {e}, recreating")
                    self._save_history({})
        except Exception as e:
            print(f"ERROR in _ensure_history_file: {e}")

    def _get_history_path(self):
        """Get the path to the usage history file"""
        try:
            history_path = os.path.join(self.lora_folder, ".lora_usage_history.json")
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            return history_path
        except Exception as e:
            print(f"ERROR getting history path: {e}")
            # Fallback to a temporary location
            return "/tmp/.lora_usage_history.json"

    def _load_history(self):
        """Load the LoRA usage history from file"""
        history_path = self._get_history_path()
        try:
            if not os.path.exists(history_path):
                print(f"History file {history_path} does not exist, returning empty history")
                return {}

            with open(history_path, 'r') as f:
                history = json.load(f)

            # Validate that it's a dictionary
            if not isinstance(history, dict):
                print("History file contains invalid data, returning empty history")
                return {}

            print(f"Successfully loaded history with {len(history)} entries from {history_path}")
            return history

        except json.JSONDecodeError as e:
            print(f"History file is corrupted (JSON error): {e}, returning empty history")
            return {}
        except Exception as e:
            print(f"ERROR loading history: {e}, returning empty history")
            return {}

    def _save_history(self, history):
        """Save the LoRA usage history to file"""
        history_path = self._get_history_path()
        try:
            # Validate input
            if not isinstance(history, dict):
                print(f"ERROR: Cannot save non-dict history: {type(history)}")
                return False

            # Create backup of existing file
            if os.path.exists(history_path):
                backup_path = history_path + ".backup"
                try:
                    import shutil
                    shutil.copy2(history_path, backup_path)
                except Exception as e:
                    print(f"Could not create backup: {e}")

            # Write new history
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)

            # Verify the file was written correctly
            with open(history_path, 'r') as f:
                test_load = json.load(f)

            if len(test_load) != len(history):
                raise Exception(f"History verification failed: expected {len(history)} entries, got {len(test_load)}")

            print(f"Successfully saved history with {len(history)} entries to {history_path}")
            return True

        except Exception as e:
            print(f"ERROR saving history to {history_path}: {e}")

            # Try to restore backup if it exists
            backup_path = history_path + ".backup"
            if os.path.exists(backup_path):
                try:
                    import shutil
                    shutil.copy2(backup_path, history_path)
                    print("Restored history from backup")
                except Exception as backup_error:
                    print(f"Could not restore backup: {backup_error}")

            return False

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

    def _get_actual_used_space(self):
        """Calculate actual used space in the volume folder (workspace/network-volume)"""
        import subprocess
        volume_root = self._get_volume_root()

        # Use du -sb for bytes output (more reliable than -sh for parsing)
        result = subprocess.run(
            ['du', '-sb', volume_root],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Output format: "12345678\t/path/to/folder"
        size_str = result.stdout.split()[0]
        total_size = int(size_str)
        print(f"du command result: {total_size / (1024 ** 3):.2f}GB for {volume_root}")
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
        try:
            print("Starting LoRA cleanup process...")

            history = self._load_history()
            print(f"Loaded history with {len(history)} entries")

            # Get all existing LoRA files (excluding hidden files and the history file)
            try:
                lora_files = [f for f in os.listdir(self.lora_folder)
                              if os.path.isfile(os.path.join(self.lora_folder, f))
                              and not f.startswith('.')
                              and f.endswith('.safetensors')]  # Only safetensors files
            except Exception as e:
                print(f"ERROR: Could not list LoRA folder contents: {e}")
                return False

            print(f"Found {len(lora_files)} LoRA files in folder")

            # If no files at all, nothing to delete
            if not lora_files:
                print("No LoRA files available for deletion")
                return False

            # For files not in history, add them with their file modification time
            current_time = time.time()
            files_added_to_history = 0

            print("Adding missing files to history...")
            for i, lora_file in enumerate(lora_files):
                try:
                    if lora_file not in history:
                        file_path = os.path.join(self.lora_folder, lora_file)
                        # Use file modification time
                        mod_time = os.path.getmtime(file_path)
                        history[lora_file] = mod_time
                        files_added_to_history += 1

                        # Print progress every 20 files to avoid log spam
                        if files_added_to_history % 20 == 0:
                            print(f"Added {files_added_to_history} files to history so far...")

                except Exception as e:
                    print(f"ERROR adding {lora_file} to history: {e}")
                    # Use a very old timestamp to prioritize for deletion
                    history[lora_file] = current_time - (365 * 24 * 3600)  # 1 year ago
                    files_added_to_history += 1

            print(f"Finished adding files to history. Added {files_added_to_history} new entries.")

            # Save the updated history if files were added
            if files_added_to_history > 0:
                try:
                    self._save_history(history)
                    print(f"Successfully saved history with {len(history)} total entries")
                except Exception as e:
                    print(f"ERROR saving history: {e}")
                    return False

            # Filter history to only include existing files
            valid_history = {k: v for k, v in history.items() if k in lora_files}
            print(f"Valid history entries: {len(valid_history)}")

            if not valid_history:
                print("ERROR: No valid history entries found")
                return False

            # Find least recently used LoRA
            try:
                least_recent_lora = min(valid_history.items(), key=lambda x: x[1])
                least_recent_file = least_recent_lora[0]
                last_used_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                               time.localtime(least_recent_lora[1]))

                print(f"Selected for deletion: {least_recent_file} (last used: {last_used_time})")

                # NEVER delete files containing "Lightning" (case insensitive)
                if "lightning" in least_recent_file.lower():
                    print(f"SKIPPING deletion: {least_recent_file} contains 'Lightning' and is protected")
                    # Remove this file from consideration and try the next one
                    valid_history.pop(least_recent_file, None)
                    if valid_history:
                        print("Looking for next least recently used file...")
                        return self._delete_least_recently_used_lora()
                    else:
                        print("No more files available for deletion (all remaining files are protected)")
                        return False
            except Exception as e:
                print(f"ERROR finding least recent file: {e}")
                return False

            # Delete the file
            try:
                lora_path = os.path.join(self.lora_folder, least_recent_file)

                # Check if file exists before trying to delete
                if not os.path.exists(lora_path):
                    print(f"File {least_recent_file} no longer exists, removing from history")
                    history.pop(least_recent_file, None)
                    self._save_history(history)
                    # Try again with remaining files
                    return self._delete_least_recently_used_lora()

                file_size = os.path.getsize(lora_path) / (1024 * 1024)  # Size in MB
                print(f"Attempting to delete: {lora_path} ({file_size:.2f} MB)")

                os.remove(lora_path)
                print(f"Successfully deleted LoRA: {least_recent_file} ({file_size:.2f} MB)")

                # Update history by removing the deleted file
                history.pop(least_recent_file, None)
                self._save_history(history)

                # Verify file was actually deleted
                if os.path.exists(lora_path):
                    print(f"ERROR: File {least_recent_file} still exists after deletion attempt!")
                    return False
                else:
                    print(f"Confirmed: File {least_recent_file} has been deleted successfully")
                    return True

            except PermissionError as e:
                print(f"PERMISSION ERROR deleting {least_recent_file}: {e}")
                return False
            except Exception as e:
                print(f"ERROR deleting LoRA file {least_recent_file}: {e}")
                return False

        except Exception as e:
            print(f"UNEXPECTED ERROR in _delete_least_recently_used_lora: {e}")
            import traceback
            traceback.print_exc()
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

            # Check available disk space before downloading
            free_space, is_reliable = self._check_disk_space()

            # Set minimum free space based on disk reliability
            # Unreliable (network volume): keep 10 GB free
            # Reliable (normal disk): keep 5 GB free
            MIN_FREE_SPACE = (5 * 1024 ** 3) if is_reliable else (10 * 1024 ** 3)  # 5GB or 10GB in bytes

            # Check volume size (new method for mounted filesystems)
            current_volume_size, volume_root = self._check_volume_size()

            # Determine if we need to free up space based on either check
            need_cleanup = False
            cleanup_reason = ""

            # Check traditional disk space
            if free_space < MIN_FREE_SPACE and free_space > 0:  # free_space > 0 means the check worked
                need_cleanup = True
                cleanup_reason = f"Low disk space: {free_space / (1024 ** 3):.2f}GB available. Need at least {MIN_FREE_SPACE / (1024 ** 3):.0f}GB."

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
                    free_space, is_reliable = self._check_disk_space()
                    # Update MIN_FREE_SPACE in case reliability changed
                    MIN_FREE_SPACE = (5 * 1024 ** 3) if is_reliable else (10 * 1024 ** 3)
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

                final_free_space, is_reliable = self._check_disk_space()
                MIN_FREE_SPACE = (5 * 1024 ** 3) if is_reliable else (10 * 1024 ** 3)
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