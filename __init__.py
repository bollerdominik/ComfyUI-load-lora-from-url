from .nodes.load_lora_url_node import LoadLoraByUrlOrPath
from .nodes.load_video_lora_url_node import LoadVideoLoraByUrlOrPath, LoadVideoLoraByUrlOrPathSelect
from .nodes.load_video_url_node import LoadVideoByUrlOrPath
from .nodes.load_upscale_model_url_node import LoadUpscaleModelByUrlOrPath
from .nodes.gemini_image2_genai_node import GeminiImage2GenAI

NODE_CLASS_MAPPINGS = {
    "LoadLoraFromUrlOrPath": LoadLoraByUrlOrPath,
    "LoadVideoLoraFromUrlOrPath": LoadVideoLoraByUrlOrPath,
    "LoadVideoLoraFromUrlOrPathSelect": LoadVideoLoraByUrlOrPathSelect,
    "LoadVideoFromUrlOrPath": LoadVideoByUrlOrPath,
    "LoadUpscaleModelFromUrlOrPath": LoadUpscaleModelByUrlOrPath,
    "GeminiImage2GenAI": GeminiImage2GenAI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLoraFromUrlOrPath": "Load LoRA (URL/Path) to Stack",
    "LoadVideoLoraFromUrlOrPath": "Load Video LoRA (URL/Path) & Apply",
    "LoadVideoLoraFromUrlOrPathSelect": "Load Video LoRA (URL/Path) Wan",
    "LoadVideoFromUrlOrPath": "Load Video (URL/Path)",
    "LoadUpscaleModelFromUrlOrPath": "Load Upscale Model (URL/Path)",
    "GeminiImage2GenAI": "Nano Banana Pro (Google Gemini Image - Python)",
}

# Conditionally load nunchaku node if module is available
try:
    import nunchaku
    from .nodes.load_nunchaku_lora_url_node import LoadNunchakuLoraFromUrlOrPath
    NODE_CLASS_MAPPINGS["LoadNunchakuLoraFromUrlOrPath"] = LoadNunchakuLoraFromUrlOrPath
    NODE_DISPLAY_NAME_MAPPINGS["LoadNunchakuLoraFromUrlOrPath"] = "Nunchaku FLUX LoRA Loader (URL/Path)"
except ImportError:
    pass
