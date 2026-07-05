from .nodes.load_lora_url_node import LoadLoraByUrlOrPath
from .nodes.load_video_lora_url_node import LoadVideoLoraByUrlOrPath, LoadVideoLoraByUrlOrPathSelect
from .nodes.load_upscale_model_url_node import LoadUpscaleModelByUrlOrPath
from .nodes.gemini_image2_genai_node import GeminiImage2GenAI
from .nodes.openrouter_gemini_image_node import OpenRouterGeminiImage
from .nodes.paste_image_by_mask_node import PasteImageByMask
from .nodes.cut_image_by_mask_node import CutImageByMask
from .nodes.image_resize_node import ImageResize
from .nodes.simple_math_node import SimpleMath
from .nodes.assert_not_black_node import AssertNotBlack
from .nodes.byteplus_video_generation_node import BytePlusVideoGeneration
from .nodes.square_mask_region_node import SquareMaskRegion

NODE_CLASS_MAPPINGS = {
    "LoadLoraFromUrlOrPath": LoadLoraByUrlOrPath,
    "LoadVideoLoraFromUrlOrPath": LoadVideoLoraByUrlOrPath,
    "LoadVideoLoraFromUrlOrPathSelect": LoadVideoLoraByUrlOrPathSelect,
    "LoadUpscaleModelFromUrlOrPath": LoadUpscaleModelByUrlOrPath,
    "GeminiImage2GenAI": GeminiImage2GenAI,
    "OpenRouterGeminiImage": OpenRouterGeminiImage,
    "PasteImageByMask": PasteImageByMask,
    "CutImageByMask": CutImageByMask,
    "ImageResize+": ImageResize,
    "SimpleMath+": SimpleMath,
    "AssertNotBlack": AssertNotBlack,
    "BytePlusVideoGeneration": BytePlusVideoGeneration,
    "SquareMaskRegion": SquareMaskRegion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLoraFromUrlOrPath": "Load LoRA (URL/Path) to Stack",
    "LoadVideoLoraFromUrlOrPath": "Load Video LoRA (URL/Path) & Apply",
    "LoadVideoLoraFromUrlOrPathSelect": "Load Video LoRA (URL/Path) Wan",
    "LoadUpscaleModelFromUrlOrPath": "Load Upscale Model (URL/Path)",
    "GeminiImage2GenAI": "Nano Banana Pro (Google Gemini Image - Python)",
    "OpenRouterGeminiImage": "Nano Banana Pro (OpenRouter Gemini Image)",
    "PasteImageByMask": "Paste Image By Mask",
    "CutImageByMask": "Cut Image By Mask",
    "ImageResize+": "🔧 Image Resize",
    "SimpleMath+": "🔧 Simple Math",
    "AssertNotBlack": "Assert Not Black",
    "BytePlusVideoGeneration": "BytePlus Video Generation",
    "SquareMaskRegion": "Square Mask Region",
}

# Conditionally load nunchaku node if module is available
try:
    import nunchaku
    from .nodes.load_nunchaku_lora_url_node import LoadNunchakuLoraFromUrlOrPath
    NODE_CLASS_MAPPINGS["LoadNunchakuLoraFromUrlOrPath"] = LoadNunchakuLoraFromUrlOrPath
    NODE_DISPLAY_NAME_MAPPINGS["LoadNunchakuLoraFromUrlOrPath"] = "Nunchaku FLUX LoRA Loader (URL/Path)"
except ImportError:
    pass

# Conditionally load video URL node if cv2 (opencv) and av are available
try:
    import cv2  # noqa: F401
    import av  # noqa: F401
    from .nodes.load_video_url_node import LoadVideoByUrlOrPath
    NODE_CLASS_MAPPINGS["LoadVideoFromUrlOrPath"] = LoadVideoByUrlOrPath
    NODE_DISPLAY_NAME_MAPPINGS["LoadVideoFromUrlOrPath"] = "Load Video (URL/Path)"
except ImportError:
    pass
