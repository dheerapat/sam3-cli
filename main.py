import argparse
import io
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlparse

import av
import matplotlib
import numpy as np
import requests
import torch
from huggingface_hub.errors import GatedRepoError
from PIL import Image
from transformers import Sam3Model, Sam3Processor, Sam3VideoModel, Sam3VideoProcessor

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def _open_video(video_input):
    if is_url(video_input):
        return av.open(video_input)
    return av.open(str(Path(video_input).expanduser().resolve()))


def load_video_frames(video_input, start_frame=0, max_frames=50):
    try:
        container = _open_video(video_input)
    except Exception as exc:
        raise SystemExit(f"Failed to open video '{video_input}': {exc}") from exc

    stream = container.streams.video[0]
    fps = stream.average_rate if stream.average_rate else Fraction(25)

    frames = []
    for idx, frame in enumerate(container.decode(stream)):
        if idx < start_frame:
            continue
        if len(frames) >= max_frames:
            break
        frames.append(frame.to_image().convert("RGB"))
        if len(frames) == 1:
            print(
                f"Decoding frames {start_frame}..{start_frame + max_frames - 1}...",
                flush=True,
            )
        if len(frames) % 10 == 0:
            print(f"  Decoded {len(frames)}/{max_frames} frames", flush=True)

    container.close()

    if not frames:
        raise SystemExit(
            f"No frames decoded (start_frame={start_frame}, video may be too short)"
        )

    print(
        f"Loaded {len(frames)} frames [{start_frame}..{start_frame + len(frames) - 1}]",
        flush=True,
    )
    return frames, fps


def default_video_output_path(video_input, start_frame=0, max_frames=50):
    if is_url(video_input):
        parsed = urlparse(video_input)
        name = Path(parsed.path).name or "video"
        stem = Path(name).stem
    else:
        stem = Path(video_input).expanduser().resolve().stem

    if start_frame > 0:
        end_frame = start_frame + max_frames - 1
        return Path.cwd() / f"{stem}.sam.f{start_frame}-{end_frame}.mp4"
    return Path.cwd() / f"{stem}.sam.mp4"


def load_video_model(device):
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    try:
        model = Sam3VideoModel.from_pretrained(
            "facebook/sam3", torch_dtype=torch_dtype, device_map=device
        )
    except GatedRepoError:
        raise SystemExit(
            "Access to facebook/sam3 is restricted.\n"
            "Accept the terms at https://huggingface.co/facebook/sam3 "
            "and authenticate with `huggingface-cli login` before running."
        )
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    return model, processor


def run_video_segmentation(model, processor, frames, prompt, device, max_frames):
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Initializing video session for {len(frames)} frames...", flush=True)
    inference_session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch_dtype,
    )
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=prompt,
    )

    print(f"Propagating through up to {max_frames} frames...", flush=True)
    outputs_per_frame = {}
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session,
        max_frame_num_to_track=max_frames,
    ):
        processed = processor.postprocess_outputs(inference_session, model_outputs)
        outputs_per_frame[model_outputs.frame_idx] = processed
        n_done = len(outputs_per_frame)
        if n_done % 5 == 0 or n_done == max_frames:
            print(f"  Propagated {n_done}/{max_frames} frames", flush=True)

    print(f"Propagation complete ({len(outputs_per_frame)} frames)", flush=True)

    return outputs_per_frame


def save_video_output(frames, outputs_per_frame, output_path, fps):
    """Composite mask overlays onto each frame and write an MP4 with PyAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = frames[0].size
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    # Reasonable quality for a demo output
    stream.options = {"crf": "18"}

    total = len(frames)
    total_objects = 0
    for idx, frame_pil in enumerate(frames):
        frame_data = outputs_per_frame.get(idx)
        if frame_data is not None and len(frame_data["masks"]) > 0:
            composited = overlay_masks(frame_pil, frame_data["masks"]).convert("RGB")
            total_objects = max(total_objects, len(frame_data["masks"]))
        else:
            composited = frame_pil.convert("RGB")

        av_frame = av.VideoFrame.from_ndarray(
            np.array(composited, dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(av_frame):
            container.mux(packet)

        if (idx + 1) % 10 == 0 or idx + 1 == total:
            print(f"  Encoded {idx + 1}/{total} frames", flush=True)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    print(f"Processed {len(outputs_per_frame)} frames", flush=True)
    print(f"Max objects detected in a single frame: {total_objects}", flush=True)
    print(f"Saved overlay video to {output_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common_args(parser):
    """Add arguments shared by both image and video subcommands."""
    parser.add_argument("-i", "--input", required=True, help="Path or URL to the input")
    parser.add_argument("-p", "--prompt", required=True, help="Text prompt to segment")
    parser.add_argument(
        "-o", "--output", type=Path, help="Path to write the output file"
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


def build_parser():
    parser = argparse.ArgumentParser(
        prog="sam",
        description="Run SAM3 text-prompted segmentation on images or videos.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="COMMAND")
    subparsers.required = True

    # -- image subcommand --
    image_parser = subparsers.add_parser(
        "image",
        help="Segment a single image",
        description="Run SAM3 text-prompted segmentation on a local image or image URL.",
    )
    _add_common_args(image_parser)

    # -- video subcommand --
    video_parser = subparsers.add_parser(
        "video",
        help="Segment and track objects in a video",
        description="Run SAM3 text-prompted segmentation and tracking on a local video or video URL.",
    )
    _add_common_args(video_parser)
    video_parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Maximum number of frames to process (default: 50)",
    )
    video_parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Frame index to start processing from (default: 0)",
    )

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


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


def run_video_command(args):
    device = get_device(args.device)
    start_frame = args.start_frame
    max_frames = args.max_frames

    print(f"Device: {device}", flush=True)
    if device == "cpu":
        print("Warning: Running on CPU — inference will be slow.", flush=True)

    output_path = (
        (args.output or default_video_output_path(args.input, start_frame, max_frames))
        .expanduser()
        .resolve()
    )

    frames, fps = load_video_frames(args.input, start_frame, max_frames)
    model, processor = load_video_model(device)
    outputs_per_frame = run_video_segmentation(
        model,
        processor,
        frames,
        args.prompt,
        device,
        max_frames,
    )

    save_video_output(frames, outputs_per_frame, output_path, fps)


def main():
    args = build_parser().parse_args()
    if args.subcommand == "image":
        run_image_command(args)
    elif args.subcommand == "video":
        run_video_command(args)


if __name__ == "__main__":
    main()
