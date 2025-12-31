import torch
import torch.nn.functional as F


def _ensure_batch(mask, batch_size):
    if mask.shape[0] == batch_size:
        return mask
    if mask.shape[0] == 1:
        return mask.repeat(batch_size, 1, 1)
    if batch_size % mask.shape[0] == 0:
        return mask.repeat(batch_size // mask.shape[0], 1, 1)
    raise ValueError("Batch size mismatch between inputs.")


def _normalize_mask(mask):
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 3:
        return mask
    if mask.ndim == 4:
        if mask.shape[1] == 1:
            return mask[:, 0, :, :]
        if mask.shape[-1] == 1:
            return mask[:, :, :, 0]
        return torch.max(mask, dim=1).values
    raise ValueError("Unsupported MASK shape.")


def _to_bhwc(image):
    if image.ndim == 2:
        return image.unsqueeze(0).unsqueeze(-1)
    if image.ndim == 4:
        if image.shape[-1] in (1, 3, 4):
            return image
        if image.shape[1] in (1, 3, 4):
            return image.permute(0, 2, 3, 1)
    raise ValueError("Expected image tensor with shape [H,W], [B,H,W,C], or [B,C,H,W].")


class CutImageByMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/compose"

    def execute(self, image, mask):
        image = _to_bhwc(image)
        if image.ndim != 4:
            raise ValueError("Expected IMAGE tensor with shape [B,H,W,C].")

        mask = _normalize_mask(mask).to(image.device)
        mask = torch.clamp(mask, 0.0, 1.0)

        batch = image.shape[0]
        mask = _ensure_batch(mask, batch)

        _, height, width, _ = image.shape
        if mask.shape[1] != height or mask.shape[2] != width:
            mask = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="nearest")[:, 0, :, :]

        hard_mask = mask > 0
        if not torch.any(hard_mask):
            return (image,)

        mask_any = torch.any(hard_mask, dim=0)
        ys, xs = torch.nonzero(mask_any, as_tuple=True)
        ymin = int(ys.min().item())
        ymax = int(ys.max().item())
        xmin = int(xs.min().item())
        xmax = int(xs.max().item())

        cropped = image[:, ymin : ymax + 1, xmin : xmax + 1, :]
        return (cropped,)
