from urllib.parse import urlparse

import matplotlib
import numpy as np
import torch
from PIL import Image


def overlay_masks(image, masks):
    image = image.convert("RGBA")
    # Use .float() before astype to correctly handle both bool and float tensors
    masks = (255 * masks.float().cpu().numpy()).astype(np.uint8)

    n_masks = masks.shape[0]
    if n_masks == 0:
        print("No masks to overlay.")
        return image

    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    for mask_array, color in zip(masks, colors):
        # Use NumPy for alpha computation instead of a slow per-pixel Python lambda
        alpha = Image.fromarray((mask_array * 0.5).astype(np.uint8))
        overlay = Image.new("RGBA", image.size, color + (0,))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image


def get_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but no CUDA device is available.")
    return device


def is_url(value):
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
