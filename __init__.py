from .nodes.load_lora_url_node import LoadLoraByUrlOrPath
from .nodes.load_video_lora_url_node import LoadVideoLoraByUrlOrPath

NODE_CLASS_MAPPINGS = {
    "LoadLoraFromUrlOrPath": LoadLoraByUrlOrPath,
    "LoadVideoLoraFromUrlOrPath": LoadVideoLoraByUrlOrPath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLoraFromUrlOrPath": "Load LoRA (URL/Path) to Stack",
    "LoadVideoLoraFromUrlOrPath": "Load Video LoRA (URL/Path) & Apply",
}
