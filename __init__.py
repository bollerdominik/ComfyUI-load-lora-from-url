from .nodes.load_lora_url_node import LoadLoraByUrlOrPath
from .nodes.load_video_lora_url_node import LoadVideoLoraByUrlOrPath, LoadVideoLoraByUrlOrPathSelect
from .nodes.load_video_url_node import LoadVideoByUrlOrPath

NODE_CLASS_MAPPINGS = {
    "LoadLoraFromUrlOrPath": LoadLoraByUrlOrPath,
    "LoadVideoLoraFromUrlOrPath": LoadVideoLoraByUrlOrPath,
    "LoadVideoLoraFromUrlOrPathSelect": LoadVideoLoraByUrlOrPathSelect,
    "LoadVideoFromUrlOrPath": LoadVideoByUrlOrPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLoraFromUrlOrPath": "Load LoRA (URL/Path) to Stack",
    "LoadVideoLoraFromUrlOrPath": "Load Video LoRA (URL/Path) & Apply",
    "LoadVideoLoraFromUrlOrPathSelect": "Load Video LoRA (URL/Path) Wan",
    "LoadVideoFromUrlOrPath": "Load Video (URL/Path)",
}

# Conditionally load nunchaku node if module is available
try:
    import nunchaku
    from .nodes.load_nunchaku_lora_url_node import LoadNunchakuLoraFromUrlOrPath
    NODE_CLASS_MAPPINGS["LoadNunchakuLoraFromUrlOrPath"] = LoadNunchakuLoraFromUrlOrPath
    NODE_DISPLAY_NAME_MAPPINGS["LoadNunchakuLoraFromUrlOrPath"] = "Nunchaku FLUX LoRA Loader (URL/Path)"
except ImportError:
    pass
