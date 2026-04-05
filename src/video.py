from fractions import Fraction
from pathlib import Path
from urllib.parse import urlparse

import av
import numpy as np
import torch
from huggingface_hub.errors import GatedRepoError
from transformers import Sam3VideoModel, Sam3VideoProcessor

from src.utils import get_device, is_url, overlay_masks


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


def count_frames(video_input):
    """Return the total number of frames in a video without decoding pixel data."""
    try:
        container = _open_video(video_input)
    except Exception as exc:
        raise SystemExit(f"Failed to open video '{video_input}': {exc}") from exc

    stream = container.streams.video[0]
    # Use the container-reported frame count when available (fast path)
    total = stream.frames
    if not total:
        # Fall back to counting by iterating packets (no pixel decode needed)
        total = sum(1 for _ in container.demux(stream) if _.pts is not None)
    container.close()
    return total


def run_video_command(args):
    if args.frames:
        total = count_frames(args.input)
        print(total)
        return

    if not args.prompt:
        raise SystemExit("error: -p/--prompt is required for segmentation")

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
