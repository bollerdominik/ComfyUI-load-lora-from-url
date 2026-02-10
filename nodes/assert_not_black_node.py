import torch


class AssertNotBlack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "image/validation"
    OUTPUT_NODE = True

    def execute(self, image):
        if torch.all(image == 0):
            raise ValueError("Image is entirely black - execution failed")
        return {}
