import base64
from io import BytesIO
from urllib.request import urlopen

import numpy as np
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


def _append_gemini_file_parts(contents, files):
    if not files:
        return
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
                contents.append(image)
            except Exception:
                continue
        elif mime_type and str(mime_type) == "text/plain":
            try:
                contents.append(data_bytes.decode("utf-8"))
            except Exception:
                continue


def _decode_image_part(part):
    inline_data = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
    if inline_data and getattr(inline_data, "data", None):
        data = inline_data.data
        if isinstance(data, str):
            data = base64.b64decode(data)
        return Image.open(BytesIO(data)).convert("RGB")

    file_data = getattr(part, "file_data", None) or getattr(part, "fileData", None)
    if file_data:
        file_uri = getattr(file_data, "file_uri", None) or getattr(file_data, "fileUri", None)
        if file_uri:
            with urlopen(file_uri) as response:
                return Image.open(BytesIO(response.read())).convert("RGB")
    return None


def _build_genai_config(seed, response_modalities, aspect_ratio, resolution, model):
    try:
        from google.genai import types as genai_types
    except Exception:
        return None

    config_kwargs = {}
    if response_modalities:
        config_kwargs["response_modalities"] = (
            ["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]
        )
    if seed is not None:
        # GenAI expects a 32-bit signed int for seed.
        config_kwargs["seed"] = int(seed) % (2**31 - 1)

    if aspect_ratio != "auto" or (resolution and model != "gemini-2.5-flash-image"):
        image_config_kwargs = {}
        if aspect_ratio != "auto":
            image_config_kwargs["aspect_ratio"] = aspect_ratio
        if resolution and model != "gemini-2.5-flash-image":
            image_config_kwargs["image_size"] = resolution
        if image_config_kwargs and hasattr(genai_types, "ImageConfig"):
            try:
                config_kwargs["image_config"] = genai_types.ImageConfig(**image_config_kwargs)
            except Exception:
                pass

    if not config_kwargs:
        return None

    try:
        return genai_types.GenerateContentConfig(**config_kwargs)
    except Exception:
        return None


def _log_usage_metadata(response):
    usage = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
    if not usage:
        return
    prompt_tokens = getattr(usage, "prompt_token_count", None) or getattr(usage, "promptTokenCount", None)
    candidates_tokens = getattr(usage, "candidates_token_count", None) or getattr(
        usage, "candidatesTokenCount", None
    )
    total_tokens = getattr(usage, "total_token_count", None) or getattr(usage, "totalTokenCount", None)
    print(f"Gemini usage: prompt={prompt_tokens}, candidates={candidates_tokens}, total={total_tokens}")


class GeminiImage2GenAI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["gemini-3-pro-image-preview", "gemini-2.5-flash-image"],),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFF}),
                "aspect_ratio": (["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],),
                "resolution": (["1K", "2K", "4K"],),
                "response_modalities": (["IMAGE+TEXT", "IMAGE"],),
                "gemini_key": ("STRING", {"default": "", "multiline": False}),
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
    CATEGORY = "api node/image/Gemini"

    def execute(
        self,
        prompt,
        model,
        seed,
        aspect_ratio,
        resolution,
        response_modalities,
        gemini_key,
        images=None,
        files=None,
        system_prompt="",
    ):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required.")
        if not gemini_key:
            raise ValueError("Gemini API key is required.")

        try:
            from google import genai
        except Exception as exc:
            raise RuntimeError("google-genai is not installed. Install the google-genai package.") from exc

        contents = []
        if system_prompt:
            combined_prompt = f"{system_prompt.strip()}\n\n{prompt.strip()}"
        else:
            combined_prompt = prompt.strip()
        contents.append(combined_prompt)

        if images is not None:
            if _get_number_of_images(images) > 14:
                raise ValueError("The current maximum number of supported images is 14.")
            for idx in range(_get_number_of_images(images)):
                contents.append(_tensor_to_pil(images[idx : idx + 1]))

        _append_gemini_file_parts(contents, files)

        client = genai.Client(api_key=gemini_key)
        config = _build_genai_config(seed, response_modalities, aspect_ratio, resolution, model)
        if config is not None:
            response = client.models.generate_content(model=model, contents=contents, config=config)
        else:
            response = client.models.generate_content(model=model, contents=contents)

        _log_usage_metadata(response)

        parts = getattr(response, "parts", None)
        if parts is None and getattr(response, "candidates", None):
            candidate = response.candidates[0]
            parts = getattr(candidate.content, "parts", None)
        if not parts:
            raise ValueError("Gemini returned no parts in the response.")

        output_images = []
        output_texts = []
        for part in parts:
            image = _decode_image_part(part)
            if image is not None:
                output_images.append(_pil_to_tensor(image))
            if getattr(part, "text", None):
                output_texts.append(part.text)

        if not output_images:
            raise ValueError("Gemini returned no image parts. Try IMAGE+TEXT for debugging output.")

        output_text = "\n".join(output_texts).strip()
        return (torch.cat(output_images, dim=0), output_text)
