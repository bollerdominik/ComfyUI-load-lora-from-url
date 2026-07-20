import hashlib
import json
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import folder_paths
import requests
from nodes import LoraLoader  # Import LoraLoader for applying LoRAs


# Raised when a server that looked range-capable answers a ranged GET with
# something other than 206 (plain 200, or the file changed under If-Range).
# Bytes written at segment offsets can then not be trusted, so the download
# falls back to a fresh single-stream fetch.
class _RangeNotSupported(Exception):
    pass


# Shared download/cache/disk-management logic for the LoRA-from-URL nodes below.
# LoRA files are always safetensors, which lets us validate them structurally:
# the file starts with an 8-byte little-endian header length, followed by a JSON
# header whose per-tensor data_offsets describe exactly how many bytes of tensor
# data must follow. A truncated or partial download fails this check immediately.
class LoraDownloadManagerBase:
    DOWNLOAD_TIMEOUT = (10, 120)  # (connect, read) timeout in seconds
    DOWNLOAD_RETRIES = 3
    DOWNLOAD_CONNECTIONS = 4  # parallel connections for segmented downloads
    SEGMENTED_MIN_SIZE = 32 * 1024 * 1024  # segment only files at least this big
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024

    def __init__(self):
        # Initialize lora folder path
        self.lora_folder = folder_paths.get_folder_paths("loras")[0]
        self.history_file = os.path.join(self.lora_folder, "history.json")
        self.min_free_space_gb = 2  # Minimum free space in GB

        # Protected LoRAs that should never be deleted
        self.protected_keywords = ["lightning", "distilled"]  # Case insensitive matching

        # Network volume settings
        self.network_volume_path = "/workspace/network-volume"  # Path to check for network volume
        self.network_volume_free_space_threshold_gb = 400  # If free space > this, consider unreliable
        self.network_volume_max_size_gb = 270  # Maximum total size for network volumes

    def download_lora(self, url):
        """Download a LoRA file from URL or copy from local path.

        Returns the filename inside the lora folder. Raises RuntimeError if the
        file cannot be fetched or fails validation, so workflows fail loudly
        instead of silently running without the LoRA.
        """
        url = url.strip()
        # Try to get the filename from the URL
        if url.startswith('http'):
            filename = os.path.basename(url.split('?')[0])
            # If no extension or no filename, use hash
            if not filename or '.' not in filename:
                filename = f"{hashlib.md5(url.encode()).hexdigest()}.safetensors"
        else:
            # Local path
            filename = os.path.basename(url)

        # Check if file already exists and is a structurally valid safetensors file
        full_path = os.path.join(self.lora_folder, filename)
        if os.path.exists(full_path):
            if self._validate_and_cleanup_file(full_path, filename):
                print(f"LoRA file {filename} already exists and is valid, skipping download")
                self._update_lora_usage(filename)
                return filename
            print(f"Existing LoRA file {filename} was invalid and removed, re-downloading...")

        if url.startswith('http'):
            try:
                self._download_to_path(url, full_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download LoRA from {url}: {e}")
            print(f"Downloaded LoRA to {full_path}")
            self._update_lora_usage(filename)
            return filename
        else:
            if not os.path.exists(url):
                raise RuntimeError(f"Local LoRA file {url} does not exist")
            # Copy via temp file + atomic rename so a partial copy never lands
            # at the final path
            tmp_path = f"{full_path}.{uuid.uuid4().hex}.part"
            try:
                shutil.copy2(url, tmp_path)
                error = self._check_safetensors(tmp_path)
                if error:
                    raise RuntimeError(f"Local LoRA file {url} is not a valid safetensors file: {error}")
                os.replace(tmp_path, full_path)
            finally:
                self._remove_quietly(tmp_path)
            print(f"Copied LoRA from {url} to {full_path}")
            self._update_lora_usage(filename)
            return filename

    def _download_to_path(self, url, full_path):
        """Download url to a temp file, verify it, then atomically move it into place.

        Uses several parallel range requests when the server supports them
        (single connections are often per-connection throttled), and resumes
        partial data on retry instead of restarting from zero. Every ranged
        request carries If-Range with the validator captured at probe time, so
        a file that changes on the server mid-download can never be stitched
        together from mismatched pieces. The final path is only ever written
        by os.replace() of a fully validated file, so an interrupted download
        can never leave a corrupt file where the cache check would find it.
        """
        tmp_path = f"{full_path}.{uuid.uuid4().hex}.part"
        try:
            force_single = False  # set when the server turns out not to honor Range
            tmp_resumable = False  # tmp_path holds a single-stream partial download
            attempt = 0
            while True:
                attempt += 1
                try:
                    print(f"Downloading LoRA from {url} (attempt {attempt}/{self.DOWNLOAD_RETRIES})")
                    total_size, supports_ranges, validator = self._probe(url)
                    if force_single:
                        supports_ranges = False

                    if supports_ranges and total_size >= self.SEGMENTED_MIN_SIZE \
                            and self.DOWNLOAD_CONNECTIONS > 1:
                        # Segmented mode writes at fixed offsets into a
                        # preallocated file; a single-stream partial from an
                        # earlier attempt must not be mixed into that
                        if tmp_resumable:
                            self._remove_quietly(tmp_path)
                        tmp_resumable = False
                        self._download_segmented(url, tmp_path, total_size, validator)
                    else:
                        resume = tmp_resumable and supports_ranges and total_size > 0
                        if not resume:
                            self._remove_quietly(tmp_path)
                        tmp_resumable = True
                        self._download_single(url, tmp_path, total_size, resume, validator)

                    written = os.path.getsize(tmp_path)
                    if total_size and written != total_size:
                        raise IOError(f"incomplete download: got {written} of {total_size} bytes")

                    error = self._check_safetensors(tmp_path)
                    if error:
                        # Complete by size yet structurally invalid: resuming
                        # cannot fix it, so the next attempt starts fresh
                        self._remove_quietly(tmp_path)
                        tmp_resumable = False
                        raise IOError(f"downloaded file is not a valid safetensors file: {error}")

                    os.replace(tmp_path, full_path)
                    return
                except _RangeNotSupported as e:
                    # Not a network failure, so it doesn't consume a retry
                    print(f"Falling back to single-stream download: {e}")
                    self._remove_quietly(tmp_path)
                    tmp_resumable = False
                    force_single = True
                    attempt -= 1
                except Exception as e:
                    # Client errors (404, 403, ...) won't succeed on retry
                    if isinstance(e, requests.HTTPError) and e.response is not None \
                            and 400 <= e.response.status_code < 500:
                        raise
                    if attempt >= self.DOWNLOAD_RETRIES:
                        raise
                    print(f"LoRA download attempt {attempt} failed: {e}")
                    time.sleep(2 * attempt)
        finally:
            self._remove_quietly(tmp_path)

    def _probe(self, url):
        """Probe the URL with a 1-byte range request.

        Returns (total_size, supports_ranges, validator). A 206 answer proves
        the server honors Range and reports the total size via Content-Range;
        a 200 means no range support, with Content-Length as the size (0 if
        unknown). The validator (strong ETag, else Last-Modified) is sent back
        as If-Range on all later ranged requests, which makes the server serve
        the full file instead of a range if the content changed in between.
        """
        headers = {'Range': 'bytes=0-0', 'Accept-Encoding': 'identity'}
        with requests.get(url, headers=headers, stream=True, timeout=self.DOWNLOAD_TIMEOUT) as response:
            if response.status_code != 206:
                if response.status_code == 416:
                    # Server rejects range probes outright; treat as unsupported
                    return 0, False, None
                response.raise_for_status()
                return int(response.headers.get('content-length', 0) or 0), False, None
            etag = response.headers.get('etag', '')
            validator = etag if etag and not etag.startswith('W/') \
                else response.headers.get('last-modified')
            total = response.headers.get('content-range', '').rsplit('/', 1)[-1]
            return (int(total) if total.isdigit() else 0), True, validator

    def _download_single(self, url, tmp_path, total_size, resume, validator):
        """Stream the whole file over one connection. With resume=True and a
        partial single-stream download already at tmp_path, continue it with a
        Range request instead of starting over."""
        pos = 0
        if resume:
            try:
                pos = os.path.getsize(tmp_path)
            except OSError:
                pos = 0
            if not 0 < pos < total_size:
                pos = 0
        headers = {'Accept-Encoding': 'identity'}
        if pos:
            headers['Range'] = f'bytes={pos}-'
            if validator:
                headers['If-Range'] = validator
            print(f"Resuming download from byte {pos} of {total_size}")
        with requests.get(url, stream=True, timeout=self.DOWNLOAD_TIMEOUT, headers=headers) as response:
            response.raise_for_status()
            if pos:
                if response.status_code != 206:
                    # Server ignored the range (or the file changed under
                    # If-Range): the response is the full file, rewrite it
                    pos = 0
                else:
                    content_range = response.headers.get('content-range', '')
                    if not content_range.startswith(f'bytes {pos}-'):
                        raise IOError(f"server answered resume from byte {pos} with "
                                      f"mismatched Content-Range '{content_range}'")
            if not pos:
                size = total_size or int(response.headers.get('content-length', 0) or 0)
                print(f"File size: {size / (1024 * 1024):.2f} MB" if size else "File size: unknown")
            with open(tmp_path, 'ab' if pos else 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.DOWNLOAD_CHUNK_SIZE):
                    f.write(chunk)

    def _download_segmented(self, url, tmp_path, total_size, validator):
        """Download the file with several parallel range requests writing into
        a preallocated temp file. Success requires every segment to have
        written exactly its byte range, so no region can be left holding the
        preallocated zero-fill."""
        seg_size = (total_size + self.DOWNLOAD_CONNECTIONS - 1) // self.DOWNLOAD_CONNECTIONS
        segments = [(start, min(start + seg_size, total_size) - 1)
                    for start in range(0, total_size, seg_size)]
        print(f"File size: {total_size / (1024 * 1024):.2f} MB, "
              f"downloading with {len(segments)} connections")
        # Preallocate so each worker can write at its own offset
        with open(tmp_path, 'wb') as f:
            f.truncate(total_size)
        with ThreadPoolExecutor(max_workers=len(segments)) as pool:
            futures = [pool.submit(self._download_segment, url, tmp_path, start, end, validator)
                       for start, end in segments]
            errors = [f.exception() for f in futures]
        errors = [e for e in errors if e is not None]
        if errors:
            # A refused range must win so the caller can fall back safely
            for e in errors:
                if isinstance(e, _RangeNotSupported):
                    raise e
            raise errors[0]

    def _download_segment(self, url, tmp_path, start, end, validator):
        """Download bytes start..end (inclusive) to their final offsets in
        tmp_path, retrying and resuming from the last byte received. Each
        worker uses its own file handle, so no shared state is involved."""
        seg_len = end - start + 1
        written = 0
        last_error = None
        for attempt in range(1, self.DOWNLOAD_RETRIES + 1):
            try:
                headers = {'Range': f'bytes={start + written}-{end}',
                           'Accept-Encoding': 'identity'}
                if validator:
                    headers['If-Range'] = validator
                with requests.get(url, stream=True, timeout=self.DOWNLOAD_TIMEOUT,
                                  headers=headers) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise _RangeNotSupported(
                            f"got HTTP {response.status_code} instead of 206 for a range request")
                    content_range = response.headers.get('content-range', '')
                    if not content_range.startswith(f'bytes {start + written}-{end}/'):
                        raise _RangeNotSupported(
                            f"mismatched Content-Range '{content_range}' for "
                            f"requested bytes {start + written}-{end}")
                    with open(tmp_path, 'r+b') as f:
                        f.seek(start + written)
                        for chunk in response.iter_content(chunk_size=self.DOWNLOAD_CHUNK_SIZE):
                            if written + len(chunk) > seg_len:
                                raise IOError(f"server sent more data than requested "
                                              f"for bytes {start}-{end}")
                            f.write(chunk)
                            written += len(chunk)
                if written == seg_len:
                    return
                raise IOError(f"segment {start}-{end} incomplete: "
                              f"got {written} of {seg_len} bytes")
            except _RangeNotSupported:
                raise
            except requests.HTTPError as e:
                if e.response is not None and 400 <= e.response.status_code < 500:
                    raise
                last_error = e
            except Exception as e:
                last_error = e
            if attempt < self.DOWNLOAD_RETRIES:
                time.sleep(2 * attempt)
        raise IOError(f"segment {start}-{end} failed after "
                      f"{self.DOWNLOAD_RETRIES} attempts: {last_error}")

    @staticmethod
    def _check_safetensors(path):
        """Structurally validate a safetensors file. Returns None if valid,
        otherwise a string describing the problem. Only reads the header, so
        it is cheap regardless of file size."""
        try:
            file_size = os.path.getsize(path)
            if file_size < 8:
                return f"file is only {file_size} bytes"
            with open(path, 'rb') as f:
                header_len = int.from_bytes(f.read(8), 'little')
                if header_len <= 0 or 8 + header_len > file_size:
                    return f"header length {header_len} does not fit in file of {file_size} bytes"
                try:
                    header = json.loads(f.read(header_len))
                except (UnicodeDecodeError, ValueError) as e:
                    return f"header is not valid JSON ({e})"
            if not isinstance(header, dict) or not header:
                return "header is not a JSON object"
            expected_data_size = 0
            for key, info in header.items():
                if key == "__metadata__":
                    continue
                offsets = info.get("data_offsets") if isinstance(info, dict) else None
                if not offsets or len(offsets) != 2:
                    return f"tensor '{key}' has no data_offsets"
                expected_data_size = max(expected_data_size, offsets[1])
            actual_data_size = file_size - 8 - header_len
            if actual_data_size < expected_data_size:
                return (f"tensor data truncated: {actual_data_size} bytes present, "
                        f"header expects {expected_data_size}")
            return None
        except OSError as e:
            return f"could not read file ({e})"

    def _validate_and_cleanup_file(self, full_path, filename):
        """Validate a LoRA file and remove it if invalid (truncated/corrupt)"""
        error = self._check_safetensors(full_path)
        if error:
            print(f"LoRA file {filename} is invalid ({error}), removing...")
            self._remove_quietly(full_path)
            return False
        return True

    @staticmethod
    def _remove_quietly(path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            print(f"Error removing file {path}: {e}")

    def _load_history(self):
        """Load LoRA usage history from JSON file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                    print(f"Loaded LoRA history: {len(history)} entries")
                    return history
            except Exception as e:
                print(f"Error loading history file: {e}")
                return {}
        else:
            print("No history file found, starting fresh")
            return {}

    def _save_history(self, history):
        """Save LoRA usage history to JSON file (atomically, so a crash can't
        leave a half-written JSON file behind)"""
        tmp_path = f"{self.history_file}.{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp_path, 'w') as f:
                json.dump(history, f, indent=2)
            os.replace(tmp_path, self.history_file)
        except Exception as e:
            print(f"Error saving history file: {e}")
            self._remove_quietly(tmp_path)

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
        history_filename = os.path.basename(self.history_file)

        # Get all LoRA files in the folder with their timestamps
        lora_files = []
        protected_files = []

        for filename in os.listdir(self.lora_folder):
            filepath = os.path.join(self.lora_folder, filename)
            # Skip directories and the history file
            if os.path.isfile(filepath) and filename != history_filename:
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


# This ComfyUI node allows you to load LoRA files from URLs or local paths
# and apply them to a model and clip.
# It downloads the files if they are not already present in the specified folder.
# It also manages disk space by tracking usage and removing least recently used files when space is low.
class LoadVideoLoraByUrlOrPath(LoraDownloadManagerBase):
    def __init__(self):
        super().__init__()
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

    @classmethod
    def IS_CHANGED(cls, toggle, num_loras, **kwargs):
        """Return a hash of inputs to enable ComfyUI caching when inputs haven't changed"""
        # Only hash configuration parameters (not model/clip objects which change between API calls)
        hash_input = f"{toggle}_{num_loras}"

        for i in range(1, int(num_loras) + 1):
            lora_url = kwargs.get(f"lora_{i}_url", "")
            # Strip query parameters from URL (e.g., AWS signatures) to get stable identifier
            lora_url_base = lora_url.split('?')[0] if lora_url else ""
            model_strength = kwargs.get(f"lora_{i}_strength_model", 1.0)
            clip_strength = kwargs.get(f"lora_{i}_strength_clip", 1.0)
            hash_input += f"_{lora_url_base}_{model_strength}_{clip_strength}"

        # Return a hash of the inputs
        result_hash = hashlib.md5(hash_input.encode()).hexdigest()
        print(f"[IS_CHANGED] Hash: {result_hash}, Input: {hash_input}")
        return result_hash

    def load_and_apply_loras(self, model, clip, toggle, num_loras, **kwargs):
        """Load and apply LoRA files to model and clip with usage tracking"""
        if toggle in [False, None, "False"]:
            return (model, clip)

        # Process each LoRA
        for i in range(1, num_loras + 1):
            lora_url = kwargs.get(f"lora_{i}_url", "")

            if not lora_url:
                continue

            # Check disk space and remove old loras if needed
            self._manage_disk_space()

            # Download/copy the LoRA file; raises on failure so the workflow
            # fails instead of silently generating without the LoRA
            lora_name = self.download_lora(lora_url)

            # Get strength values
            model_strength = float(kwargs.get(f"lora_{i}_strength_model", 1.0))
            clip_strength = float(kwargs.get(f"lora_{i}_strength_clip", 1.0))

            # Apply the LoRA using LoraLoader
            model, clip = self.lora_loader.load_lora(model, clip, lora_name, model_strength, clip_strength)
            print(f"Applied LoRA {lora_name} with strengths - model: {model_strength}, clip: {clip_strength}")

        # Refresh the lora list to include newly downloaded files
        folder_paths.get_filename_list("loras")

        return (model, clip)


# Node for loading LoRAs into WANVIDLORA format (for Wan Video LoRA workflow)
class LoadVideoLoraByUrlOrPathSelect(LoraDownloadManagerBase):
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

    def load_and_select_loras(self, toggle, num_loras, prev_lora=None, **kwargs):
        """Load LoRA files and return as WANVIDLORA format with usage tracking"""
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

            # Download/copy the LoRA file; raises on failure so the workflow
            # fails instead of silently generating without the LoRA
            lora_filename = self.download_lora(lora_url)

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
