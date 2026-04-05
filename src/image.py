import io
from pathlib import Path
from urllib.parse import urlparse

import requests
import torch
from huggingface_hub.errors import GatedRepoError
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from src.utils import get_device, is_url, overlay_masks


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


def run_image_command(args):
    device = get_device(args.device)
    output_path = (
        (args.output or default_output_path(args.input)).expanduser().resolve()
    )

    image = load_image(args.input)
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
