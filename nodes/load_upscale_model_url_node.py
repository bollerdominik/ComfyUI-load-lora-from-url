import os
import requests
import hashlib
import folder_paths
import json
import time
import shutil
from spandrel import ModelLoader, ImageModelDescriptor
import comfy.utils

try:
    from spandrel_extra_arches import EXTRA_REGISTRY
    from spandrel import MAIN_REGISTRY
    MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    print("Successfully imported spandrel_extra_arches: support for non commercial upscale models.")
except:
    pass


class LoadUpscaleModelByUrlOrPath:
    """
    ComfyUI node that loads upscale models from URLs or local paths.
    Downloads the model if it doesn't exist yet in the upscale_models folder.
    """

    def __init__(self):
        # Initialize upscale model folder path
        self.upscale_folder = folder_paths.get_folder_paths("upscale_models")[0]
        # Ensure history file exists
        self._ensure_history_file()

        # Volume size limit in bytes (100GB)
        self.VOLUME_SIZE_LIMIT = 100 * 1024 * 1024 * 1024
        # Threshold before cleanup (90GB)
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
            history_path = os.path.join(self.upscale_folder, ".upscale_model_usage_history.json")
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            return history_path
        except Exception as e:
            print(f"ERROR getting history path: {e}")
            # Fallback to a temporary location
            import tempfile
            return os.path.join(tempfile.gettempdir(), ".upscale_model_usage_history.json")

    def _load_history(self):
        """Load the upscale model usage history from file"""
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
        """Save the upscale model usage history to file"""
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

    def _update_model_usage(self, model_name):
        """Update the last usage timestamp for an upscale model"""
        history = self._load_history()
        history[model_name] = time.time()
        self._save_history(history)

    def _check_disk_space(self):
        """Check available disk space in the upscale model folder"""
        try:
            total, used, free = shutil.disk_usage(self.upscale_folder)
            return free
        except Exception as e:
            print(f"Error checking disk space: {e}")
            return 0

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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Enter model URL or local file path"
                }),
            },
        }

    RETURN_TYPES = ("UPSCALE_MODEL",)
    RETURN_NAMES = ("upscale_model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"

    def download_model(self, url):
        """Download an upscale model from URL or copy from local path"""
        # Get filename from url or generate one from hash if no filename is present
        try:
            # Try to get the filename from the URL
            if url.startswith('http'):
                filename = os.path.basename(url.split('?')[0])
                # If no extension or no filename, use hash
                if not filename or '.' not in filename:
                    filename = f"{hashlib.md5(url.encode()).hexdigest()}.pth"
            else:
                # Local path
                filename = os.path.basename(url)

            # Check if file already exists
            full_path = os.path.join(self.upscale_folder, filename)
            if os.path.exists(full_path):
                # Validate existing file before using it
                if self._validate_and_cleanup_file(full_path, filename):
                    print(f"Upscale model {filename} already exists, skipping download")
                    # Update usage history for existing file
                    self._update_model_usage(filename)
                    return filename
                else:
                    print(f"Existing upscale model {filename} was invalid and removed, re-downloading...")

            # Check available disk space before downloading
            MIN_FREE_SPACE = 2 * 1024 * 1024 * 1024  # 2GB in bytes
            free_space = self._check_disk_space()

            if free_space > 0 and free_space < MIN_FREE_SPACE:
                print(f"WARNING: Low disk space: {free_space / (1024 ** 3):.2f}GB available")

            # Download the file
            if url.startswith('http'):
                print(f"Downloading upscale model from {url}")
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an exception for HTTP errors

                # Get file size from headers if available
                file_size = int(response.headers.get('content-length', 0)) / (1024 * 1024) if 'content-length' in response.headers else "unknown"
                print(f"File size: {file_size} MB" if isinstance(file_size, (int, float)) else "File size: unknown")

                # Save the file
                with open(full_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded upscale model to {full_path}")

                # Validate the downloaded file
                if not self._validate_and_cleanup_file(full_path, filename):
                    print(f"Downloaded upscale model {filename} was invalid and removed")
                    return None

                # Update usage history for the new file
                self._update_model_usage(filename)
                return filename
            else:
                # Copy local file if it's a valid path
                if os.path.exists(url):
                    shutil.copy2(url, full_path)
                    print(f"Copied upscale model from {url} to {full_path}")

                    # Validate the copied file
                    if not self._validate_and_cleanup_file(full_path, filename):
                        print(f"Copied upscale model {filename} was invalid and removed")
                        return None

                    # Update usage history for the new file
                    self._update_model_usage(filename)
                    return filename
                else:
                    print(f"Local upscale model file {url} does not exist")
                    return None

        except Exception as e:
            print(f"Error downloading upscale model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_model(self, url):
        """Load upscale model from URL or local path"""
        if not url or url.strip() == "":
            raise ValueError("URL or path cannot be empty")

        # Download or get the model file
        model_name = self.download_model(url.strip())

        if not model_name:
            raise Exception(f"Failed to download or access upscale model from: {url}")

        # Load the model using ComfyUI's standard method
        try:
            model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
            sd = comfy.utils.load_torch_file(model_path, safe_load=True)

            # Handle specific state dict formats
            if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
                sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})

            out = ModelLoader().load_from_state_dict(sd).eval()

            if not isinstance(out, ImageModelDescriptor):
                raise Exception("Upscale model must be a single-image model.")

            # Refresh the upscale model list to include newly downloaded files
            folder_paths.get_filename_list("upscale_models")

            return (out,)

        except Exception as e:
            print(f"Error loading upscale model: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Failed to load upscale model {model_name}: {str(e)}")
