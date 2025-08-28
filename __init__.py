from .nodes.load_lora_url_node import LoadLoraByUrlOrPath
from .nodes.load_video_lora_url_node import LoadVideoLoraByUrlOrPath, LoadVideoLoraByUrlOrPathSelect
from .nodes.load_nunchaku_lora_url_node import LoadNunchakuLoraFromUrlOrPath

NODE_CLASS_MAPPINGS = {
    "LoadLoraFromUrlOrPath": LoadLoraByUrlOrPath,
    "LoadVideoLoraFromUrlOrPath": LoadVideoLoraByUrlOrPath,
    "LoadVideoLoraFromUrlOrPathSelect": LoadVideoLoraByUrlOrPathSelect,
    "LoadNunchakuLoraFromUrlOrPath": LoadNunchakuLoraFromUrlOrPath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLoraFromUrlOrPath": "Load LoRA (URL/Path) to Stack",
    "LoadVideoLoraFromUrlOrPath": "Load Video LoRA (URL/Path) & Apply",
    "LoadVideoLoraFromUrlOrPathSelect": "Load Video LoRA (URL/Path) Wan",
    "LoadNunchakuLoraFromUrlOrPath": "Nunchaku FLUX LoRA Loader (URL/Path)"
}
