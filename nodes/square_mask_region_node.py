import torch


class SquareMaskRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK",),
            "padding": ("INT", {"default": 0, "min": 0, "max": 512}),
        }}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "mask"

    def run(self, mask, padding):
        out = torch.zeros_like(mask)
        for i in range(mask.shape[0]):
            m = mask[i]
            ys, xs = torch.where(m > 0.5)
            if len(xs) == 0:
                continue
            x0, x1 = xs.min().item(), xs.max().item()
            y0, y1 = ys.min().item(), ys.max().item()
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            half = (max(x1 - x0, y1 - y0) // 2) + padding
            H, W = m.shape
            # shift the square back inside the frame instead of clipping it
            nx0 = max(0, min(cx - half, W - 2 * half))
            ny0 = max(0, min(cy - half, H - 2 * half))
            out[i, ny0:ny0 + 2 * half, nx0:nx0 + 2 * half] = 1.0
        return (out,)
