import base64
import os
import time
import uuid
from io import BytesIO

import folder_paths
import numpy as np
import requests
from PIL import Image

try:
    from comfy_api.latest import InputImpl
except Exception:
    InputImpl = None


API_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"
TERMINAL_STATUSES = {"succeeded", "failed", "expired", "cancelled"}
ACTIVE_STATUSES = {"queued", "running"}


def _tensor_to_data_url(image):
    if image is None:
        return None
    if image.ndim == 4:
        image = image[0]

    image = image.detach().cpu()
    image_array = np.clip(image.numpy() * 255.0, 0, 255).astype(np.uint8)
    if image_array.shape[-1] == 4:
        pil_image = Image.fromarray(image_array, "RGBA").convert("RGB")
    else:
        pil_image = Image.fromarray(image_array, "RGB")

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_image}"


def _headers(api_key):
    api_key = api_key.strip()
    if not api_key:
        raise ValueError("BytePlus API key is required.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _raise_for_api_error(response, action):
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:2000]
        raise RuntimeError(f"BytePlus {action} failed: HTTP {response.status_code}: {body}") from exc


def _download_video(video_url, task_id):
    response = requests.get(video_url, timeout=300)
    _raise_for_api_error(response, "video download")

    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"byteplus_{task_id}_{uuid.uuid4().hex[:8]}.mp4"
    video_path = os.path.join(temp_dir, filename)

    with open(video_path, "wb") as video_file:
        video_file.write(response.content)

    return video_path


class BytePlusVideoGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "ep-20260423190508-bhljb", "multiline": False}),
                "text": ("STRING", {"default": "my text", "multiline": True, "dynamicPrompts": False}),
                "duration": ("INT", {"default": 10, "min": 1, "max": 60, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295, "step": 1}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_url": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "ratio": (["adaptive", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], {"default": "adaptive"}),
                "resolution": (["480p", "720p"], {"default": "480p"}),
                "watermark": ("BOOLEAN", {"default": False}),
                "poll_interval_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 0.5}),
                "timeout_seconds": ("INT", {"default": 900, "min": 30, "max": 7200, "step": 30}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "video_url")
    FUNCTION = "generate"
    CATEGORY = "image/video"

    def _create_task(self, api_key, model, text, image, image_url, duration, seed, camera_fixed, generate_audio, ratio, resolution, watermark):
        content = [{"type": "text", "text": text}]
        encoded_image_url = _tensor_to_data_url(image)
        image_url = image_url.strip()
        if encoded_image_url:
            content.append({"type": "image_url", "image_url": {"url": encoded_image_url}})
        elif image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        payload = {
            "model": model.strip(),
            "content": content,
            "generate_audio": bool(generate_audio),
            "ratio": ratio,
            "resolution": resolution,
            "duration": int(duration),
            "seed": int(seed),
            "watermark": bool(watermark),
            "camera_fixed": bool(camera_fixed),
        }

        response = requests.post(API_BASE_URL, headers=_headers(api_key), json=payload, timeout=120)
        _raise_for_api_error(response, "task creation")
        data = response.json()
        task_id = data.get("id")
        if not task_id:
            raise RuntimeError(f"BytePlus task creation did not return an id: {data}")
        return task_id

    def _poll_task(self, api_key, task_id, poll_interval_seconds, timeout_seconds):
        deadline = time.monotonic() + float(timeout_seconds)
        poll_url = f"{API_BASE_URL}/{task_id}"

        while True:
            response = requests.get(poll_url, headers=_headers(api_key), timeout=120)
            _raise_for_api_error(response, "task polling")
            data = response.json()
            status = data.get("status")

            print(f"BytePlus video task {task_id}: {status}")

            if status == "succeeded":
                video_url = (data.get("content") or {}).get("video_url")
                if not video_url:
                    raise RuntimeError(f"BytePlus task {task_id} succeeded without content.video_url: {data}")
                return video_url

            if status in {"failed", "expired", "cancelled"}:
                raise RuntimeError(f"BytePlus task {task_id} ended with status '{status}': {data}")

            if status not in ACTIVE_STATUSES:
                raise RuntimeError(f"BytePlus task {task_id} returned unknown status '{status}': {data}")

            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for BytePlus task {task_id} after {timeout_seconds} seconds.")

            time.sleep(float(poll_interval_seconds))

    def generate(
        self,
        api_key,
        model,
        text,
        duration,
        seed,
        camera_fixed,
        image=None,
        image_url="",
        generate_audio=True,
        ratio="adaptive",
        resolution="480p",
        watermark=False,
        poll_interval_seconds=5.0,
        timeout_seconds=900,
    ):
        if InputImpl is None:
            raise RuntimeError("ComfyUI VIDEO output support is unavailable in this ComfyUI install.")

        task_id = self._create_task(
            api_key,
            model,
            text,
            image,
            image_url,
            duration,
            seed,
            camera_fixed,
            generate_audio,
            ratio,
            resolution,
            watermark,
        )
        video_url = self._poll_task(api_key, task_id, poll_interval_seconds, timeout_seconds)
        video_path = _download_video(video_url, task_id)
        return (InputImpl.VideoFromFile(video_path), task_id, video_url)
