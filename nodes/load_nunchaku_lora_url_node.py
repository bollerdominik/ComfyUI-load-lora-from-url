import copy
import logging
import os
import requests
import hashlib
import folder_paths
import json
import time
import shutil
from pathlib import Path

from nunchaku.lora.flux import to_diffusers

# Import the ComfyFluxWrapper - adjust path if needed
ComfyFluxWrapper = None
try:
    # Try to import from the ComfyUI-nunchaku custom node
    import sys
    import os
    
    # Get the custom_nodes directory path
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    custom_nodes_dir = os.path.dirname(current_dir)
    nunchaku_path = os.path.join(custom_nodes_dir, "ComfyUI-nunchaku")
    
    if os.path.exists(nunchaku_path):
        sys.path.insert(0, nunchaku_path)
        from wrappers.flux import ComfyFluxWrapper
        print("Successfully imported ComfyFluxWrapper from ComfyUI-nunchaku")
    else:
        print("ComfyUI-nunchaku directory not found, trying alternative imports...")
        raise ImportError("ComfyUI-nunchaku not found")
        
except ImportError:
    try:
        # Alternative import approaches
        from ComfyUI_nunchaku.wrappers.flux import ComfyFluxWrapper
        print("Successfully imported ComfyFluxWrapper from ComfyUI_nunchaku")
    except ImportError:
        try:
            # Try direct import if it's in the path
            from wrappers.flux import ComfyFluxWrapper
            print("Successfully imported ComfyFluxWrapper directly")
        except ImportError:
            print("Warning: Could not import ComfyFluxWrapper. Make sure ComfyUI-nunchaku is installed.")
            ComfyFluxWrapper = None

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LoadNunchakuLoraFromUrlOrPath:
    """
    Node for loading LoRA files from URLs or paths and applying them to Nunchaku FLUX models.
    Combines URL/path loading with caching and Nunchaku-specific LoRA conversion.
    """
    
    def __init__(self):
        # Initialize lora folder path
        self.lora_folder = folder_paths.get_folder_paths("loras")[0]
        # Ensure history file exists
        self._ensure_history_file()

        # Volume size limit in bytes (100GB = 100 * 1024^3)
        self.VOLUME_SIZE_LIMIT = 100 * 1024 * 1024 * 1024
        # Threshold before cleanup (90GB = 90 * 1024^3)
        self.VOLUME_CLEANUP_THRESHOLD = 90 * 1024 * 1024 * 1024

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
            history_path = os.path.join(self.lora_folder, ".nunchaku_lora_usage_history.json")
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            return history_path
        except Exception as e:
            print(f"ERROR getting history path: {e}")
            # Fallback to a temporary location
            return "/tmp/.nunchaku_lora_usage_history.json"

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

        while current_path != '/' and current_path != '\\' and len(current_path) > 3:  # Windows compatibility
            folder_name = os.path.basename(current_path)
            if any(indicator in folder_name for indicator in volume_indicators):
                return current_path
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:  # Reached root
                break
            current_path = parent_path

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
            for lora_file in lora_files:
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

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types and tooltips for the node.
        """
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model the LoRA will be applied to. "
                        "Make sure the model is loaded by `Nunchaku FLUX DiT Loader`."
                    },
                ),
                "lora_url": (
                    "STRING", 
                    {
                        "default": "", 
                        "multiline": True,
                        "tooltip": "URL or local path to the LoRA file to download/load."
                    }
                ),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"
    TITLE = "Nunchaku FLUX LoRA Loader (URL/Path)"

    CATEGORY = "Nunchaku"
    DESCRIPTION = (
        "LoRAs are used to modify the diffusion model, "
        "altering the way in which latents are denoised such as applying styles. "
        "This node loads LoRAs from URLs or local paths with caching and applies Nunchaku-specific conversion."
    )

    def load_lora(self, model, lora_url: str, lora_strength: float):
        """
        Apply a LoRA to a Nunchaku FLUX diffusion model with URL/path loading and caching.

        Parameters
        ----------
        model : object
            The diffusion model to modify.
        lora_url : str
            The URL or path of the LoRA to download/load and apply.
        lora_strength : float
            The strength with which to apply the LoRA.

        Returns
        -------
        tuple
            A tuple containing the modified diffusion model.
        """
        if not lora_url or not lora_url.strip():
            return (model,)  # Return original model if no URL provided

        if abs(lora_strength) < 1e-5:
            return (model,)  # If the strength is too small, return the original model

        # Download/copy the LoRA file with disk space management
        lora_name = self.download_lora(lora_url.strip())
        
        if not lora_name:
            print(f"Failed to download/load LoRA from {lora_url}")
            return (model,)

        print(f"Successfully loaded LoRA: {lora_name}")

        model_wrapper = model.model.diffusion_model
        
        # Check if this is a ComfyFluxWrapper (flexible check)
        wrapper_type_name = type(model_wrapper).__name__
        if wrapper_type_name != "ComfyFluxWrapper":
            print(f"Error: Expected ComfyFluxWrapper, got {wrapper_type_name}. Make sure the model is loaded by Nunchaku FLUX DiT Loader.")
            return (model,)

        transformer = model_wrapper.model
        model_wrapper.model = None
        ret_model = copy.deepcopy(model)  # copy everything except the model
        ret_model_wrapper = ret_model.model.diffusion_model
        # Flexible check for the copied model wrapper too
        if type(ret_model_wrapper).__name__ != "ComfyFluxWrapper":
            print(f"Error: Copied model wrapper is not ComfyFluxWrapper: {type(ret_model_wrapper).__name__}")
            return (model,)

        model_wrapper.model = transformer
        ret_model_wrapper.model = transformer

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        ret_model_wrapper.loras.append((lora_path, lora_strength))

        # Convert LoRA to Nunchaku format
        sd = to_diffusers(lora_path)

        # To handle FLUX.1 tools LoRAs, which change the number of input channels
        if "transformer.x_embedder.lora_A.weight" in sd:
            new_in_channels = sd["transformer.x_embedder.lora_A.weight"].shape[1]
            assert new_in_channels % 4 == 0
            new_in_channels = new_in_channels // 4

            old_in_channels = ret_model.model.model_config.unet_config["in_channels"]
            if old_in_channels < new_in_channels:
                ret_model.model.model_config.unet_config["in_channels"] = new_in_channels

        # Refresh the lora list to include newly downloaded files
        folder_paths.get_filename_list("loras")

        print(f"Applied Nunchaku LoRA {lora_name} with strength {lora_strength}")
        return (ret_model,)