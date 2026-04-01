import argparse
import io
from pathlib import Path
from urllib.parse import urlparse

import matplotlib
import numpy as np
import requests
import torch
from huggingface_hub.errors import GatedRepoError
from PIL import Image
from transformers import Sam3Model, Sam3Processor


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


def load_image(image_input):
    if is_url(image_input):
        try:
            response = requests.get(image_input, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SystemExit(
                f"Failed to download image '{image_input}': {exc}"
            ) from exc

        try:
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except OSError as exc:
            raise SystemExit(
                f"Failed to decode image from '{image_input}': {exc}"
            ) from exc

    image_path = Path(image_input).expanduser().resolve()
    try:
        return Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise SystemExit(f"Image not found: {image_path}")
    except OSError as exc:
        raise SystemExit(f"Failed to open image '{image_path}': {exc}") from exc


def default_output_path(image_input):
    if is_url(image_input):
        parsed = urlparse(image_input)
        name = Path(parsed.path).name or "image"
        return Path.cwd() / f"{Path(name).stem}.sam.png"

    image_path = Path(image_input).expanduser().resolve()
    return image_path.with_suffix(".sam.png")


def load_model(device):
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    try:
        model: Sam3Model = Sam3Model.from_pretrained(
            "facebook/sam3", torch_dtype=torch_dtype, device_map=device
        )
    except GatedRepoError:
        raise SystemExit(
            "Access to facebook/sam3 is restricted.\n"
            "Accept the terms at https://huggingface.co/facebook/sam3 "
            "and authenticate with `huggingface-cli login` before running."
        )
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    return model, processor


def run_segmentation(
    model, processor, image, prompt, device, threshold, mask_threshold
):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=torch.Tensor(inputs["original_sizes"]).tolist(),
    )[0]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="sam",
        description="Run SAM3 text-prompted segmentation on a local image or image URL.",
    )
    parser.add_argument(
        "-i", "--image", required=True, help="Path or URL to the input image"
    )
    parser.add_argument("-p", "--prompt", required=True, help="Text prompt to segment")
    parser.add_argument(
        "-o", "--output", type=Path, help="Path to write the overlay image"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to run inference on (default: auto)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Instance confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Mask threshold (default: 0.5)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    device = get_device(args.device)
    output_path = (
        (args.output or default_output_path(args.image)).expanduser().resolve()
    )

    image = load_image(args.image)
    model, processor = load_model(device)
    results = run_segmentation(
        model,
        processor,
        image,
        args.prompt,
        device,
        args.threshold,
        args.mask_threshold,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_masks(image, masks=results["masks"]).save(output_path)

    print(f"Found {len(results['masks'])} objects")
    print(f"Saved overlay image to {output_path}")


if __name__ == "__main__":
    main()
