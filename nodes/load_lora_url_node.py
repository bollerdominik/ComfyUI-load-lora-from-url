import os
import requests
from io import BytesIO
import hashlib
import folder_paths


class LoadLoraByUrlOrPath:
    def __init__(self):
        pass

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
            inputs["optional"][f"lora_{i}_url"] = ("STRING", {"default": "", "multiline": True})
            inputs["optional"][f"lora_{i}_strength"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["optional"][f"lora_{i}_model_strength"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["optional"][f"lora_{i}_clip_strength"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        return inputs

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "load_and_stack"

    CATEGORY = "EasyUse/Loaders"

    def download_lora(self, url, lora_folder):
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
            full_path = os.path.join(lora_folder, filename)
            if os.path.exists(full_path):
                print(f"LoRA file {filename} already exists, skipping download")
                return filename

            # Download the file
            if url.startswith('http'):
                print(f"Downloading LoRA from {url}")
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an exception for HTTP errors

                # Save the file
                with open(full_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded LoRA to {full_path}")
                return filename
            else:
                # Copy local file if it's a valid path
                if os.path.exists(url):
                    import shutil
                    shutil.copy2(url, full_path)
                    print(f"Copied LoRA from {url} to {full_path}")
                    return filename
                else:
                    print(f"Local LoRA file {url} does not exist")
                    return None

        except Exception as e:
            print(f"Error downloading LoRA: {e}")
            return None

    def load_and_stack(self, toggle, mode, num_loras, optional_lora_stack=None, **kwargs):
        if (toggle in [False, None, "False"]):
            return (None,)

        loras = []

        # Import Stack values
        if optional_lora_stack is not None:
            loras.extend([l for l in optional_lora_stack if l[0] != "None"])

        # Get the lora folder path
        lora_folder = folder_paths.get_folder_paths("loras")[0]

        # Process each LoRA
        for i in range(1, num_loras + 1):
            lora_url = kwargs.get(f"lora_{i}_url", "")

            if not lora_url:
                continue

            # Download/copy the LoRA file
            lora_name = self.download_lora(lora_url, lora_folder)

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
        folder_paths.get_filename_list("loras", force_reload=True)

        return (loras,)