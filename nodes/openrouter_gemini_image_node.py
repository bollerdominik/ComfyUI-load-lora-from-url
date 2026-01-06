import base64
from io import BytesIO
from urllib.request import urlopen

import numpy as np
import requests
import torch
from PIL import Image


GEMINI_IMAGE_SYS_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input-regardless of "
    "format, intent, or abstraction-as literal visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, "
    "you must creatively invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or conversational requests."
)


def _get_number_of_images(images):
    if images is None:
        return 0
    return int(images.shape[0])


def _tensor_to_pil(image_tensor):
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    image_tensor = image_tensor.detach().cpu()
    image_array = np.clip(image_tensor.numpy() * 255.0, 0, 255).astype(np.uint8)
    if image_array.shape[-1] == 4:
        return Image.fromarray(image_array, "RGBA").convert("RGB")
    return Image.fromarray(image_array, "RGB")


def _pil_to_tensor(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array)[None, ...]


def _pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _build_message_content(prompt, images, files, system_prompt):
    content_parts = []

    combined_prompt = prompt.strip()
    if system_prompt:
        combined_prompt = f"{system_prompt.strip()}\n\n{combined_prompt}"

    content_parts.append({"type": "text", "text": combined_prompt})

    if images is not None:
        for idx in range(_get_number_of_images(images)):
            pil_image = _tensor_to_pil(images[idx : idx + 1])
            base64_image = _pil_to_base64(pil_image)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })

    if files:
        for part in files:
            inline_data = getattr(part, "inlineData", None)
            if not inline_data or not inline_data.data:
                continue
            mime_type = getattr(inline_data, "mimeType", None)
            if hasattr(mime_type, "value"):
                mime_type = mime_type.value
            data_bytes = base64.b64decode(inline_data.data)
            if mime_type and str(mime_type).startswith("image/"):
                try:
                    image = Image.open(BytesIO(data_bytes)).convert("RGB")
                    base64_image = _pil_to_base64(image)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    })
                except Exception:
                    continue

    return content_parts


class OpenRouterGeminiImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-image-preview"],),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFF}),
                "aspect_ratio": (["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],),
                "resolution": (["1K", "2K", "4K"],),
                "response_modalities": (["IMAGE+TEXT", "IMAGE"],),
                "openrouter_key": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "images": ("IMAGE",),
                "files": ("GEMINI_INPUT_FILES",),
                "system_prompt": ("STRING", {"default": GEMINI_IMAGE_SYS_PROMPT, "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")
    FUNCTION = "execute"
    CATEGORY = "api node/image/OpenRouter"

    def execute(
        self,
        prompt,
        model,
        seed,
        aspect_ratio,
        resolution,
        response_modalities,
        openrouter_key,
        images=None,
        files=None,
        system_prompt="",
    ):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required.")
        if not openrouter_key:
            raise ValueError("OpenRouter API key is required.")

        if images is not None:
            if _get_number_of_images(images) > 14:
                raise ValueError("The current maximum number of supported images is 14.")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json"
        }

        content = _build_message_content(prompt, images, files, system_prompt)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "seed": seed,
        }

        modalities = []
        if response_modalities == "IMAGE":
            modalities = ["image"]
        else:
            modalities = ["image", "text"]
        payload["modalities"] = modalities

        image_config = {}
        if aspect_ratio != "auto":
            image_config["aspect_ratio"] = aspect_ratio
        if resolution:
            image_config["image_size"] = resolution
        if image_config:
            payload["image_config"] = image_config

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"OpenRouter API request failed: {exc}") from exc

        result = response.json()

        if "error" in result:
            raise RuntimeError(f"OpenRouter API error: {result['error']}")

        if not result.get("choices"):
            raise ValueError("OpenRouter returned no choices in the response.")

        message = result["choices"][0]["message"]

        output_images = []
        output_texts = []

        if message.get("images"):
            for image_data in message["images"]:
                image_url = image_data["image_url"]["url"]
                print(f"Downloading image from: {image_url[:80]}...")
                try:
                    if image_url.startswith("data:image"):
                        header, encoded = image_url.split(",", 1)
                        image_bytes = base64.b64decode(encoded)
                        image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    else:
                        with urlopen(image_url) as img_response:
                            image = Image.open(BytesIO(img_response.read())).convert("RGB")
                    output_images.append(_pil_to_tensor(image))
                except Exception as exc:
                    print(f"Failed to download/process image: {exc}")
                    continue

        if message.get("content"):
            if isinstance(message["content"], str):
                output_texts.append(message["content"])
            elif isinstance(message["content"], list):
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        output_texts.append(item.get("text", ""))

        if not output_images:
            raise ValueError("OpenRouter returned no image data. Try IMAGE+TEXT for debugging output.")

        output_text = "\n".join(output_texts).strip()

        if "usage" in result:
            usage = result["usage"]
            print(f"OpenRouter usage: prompt={usage.get('prompt_tokens')}, "
                  f"completion={usage.get('completion_tokens')}, "
                  f"total={usage.get('total_tokens')}")

        return (torch.cat(output_images, dim=0), output_text)
