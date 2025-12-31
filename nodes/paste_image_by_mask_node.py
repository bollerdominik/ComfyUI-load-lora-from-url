import torch
import torch.nn.functional as F


FEATHER_KERNEL = 7


def _ensure_batch(tensor, batch_size):
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.repeat(batch_size, 1, 1, 1)
    if batch_size % tensor.shape[0] == 0:
        return tensor.repeat(batch_size // tensor.shape[0], 1, 1, 1)
    raise ValueError("Batch size mismatch between inputs.")


def _normalize_mask(mask):
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 3:
        return mask
    if mask.ndim == 4:
        if mask.shape[1] == 1:
            return mask[:, 0, :, :]
        return torch.max(mask, dim=1).values
    raise ValueError("Unsupported MASK shape.")


def _match_channels(source, cropped):
    source_c = source.shape[-1]
    cropped_c = cropped.shape[-1]
    if source_c == cropped_c:
        return cropped
    if source_c == 3 and cropped_c == 4:
        return cropped[:, :, :, :3]
    if source_c == 4 and cropped_c == 3:
        alpha = torch.ones_like(cropped[:, :, :, :1])
        return torch.cat([cropped, alpha], dim=-1)
    raise ValueError("Unsupported channel configuration.")


class PasteImageByMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "cropped_image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/compose"

    def execute(self, source_image, cropped_image, mask):
        if source_image.ndim != 4 or cropped_image.ndim != 4:
            raise ValueError("Expected IMAGE tensors with shape [B,H,W,C].")

        source = source_image
        cropped = _match_channels(source, cropped_image)

        mask = _normalize_mask(mask).to(source.device)
        mask = torch.clamp(mask, 0.0, 1.0)

        batch = source.shape[0]
        source = _ensure_batch(source, batch)
        cropped = _ensure_batch(cropped, batch)
        mask = _ensure_batch(mask.unsqueeze(-1), batch).squeeze(-1)

        _, H, W, _ = source.shape
        if mask.shape[1] != H or mask.shape[2] != W:
            mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode="nearest")[:, 0, :, :]

        result = source.clone()
        for i in range(batch):
            mask_i = mask[i]
            if torch.max(mask_i) <= 0:
                continue
            hard_mask = mask_i > 0
            ys, xs = torch.nonzero(hard_mask, as_tuple=True)
            if ys.numel() == 0 or xs.numel() == 0:
                continue
            ymin = int(ys.min().item())
            ymax = int(ys.max().item())
            xmin = int(xs.min().item())
            xmax = int(xs.max().item())

            target_h = ymax - ymin + 1
            target_w = xmax - xmin + 1

            crop_i = cropped[i].permute(2, 0, 1).unsqueeze(0)
            resized = F.interpolate(crop_i, size=(target_h, target_w), mode="nearest")
            resized = resized.squeeze(0).permute(1, 2, 0)

            softened = F.avg_pool2d(
                mask_i.unsqueeze(0).unsqueeze(0),
                kernel_size=FEATHER_KERNEL,
                stride=1,
                padding=FEATHER_KERNEL // 2,
            )[0, 0]
            alpha = softened[ymin : ymax + 1, xmin : xmax + 1].unsqueeze(-1)
            base_region = result[i, ymin : ymax + 1, xmin : xmax + 1, :]
            result[i, ymin : ymax + 1, xmin : xmax + 1, :] = resized * alpha + base_region * (1.0 - alpha)

        return (result,)
