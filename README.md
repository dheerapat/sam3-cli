# SAM3-CLI 

![](/showcase.png)

Minimal CLI for running text-prompted segmentation on local images, image URLs, local videos, and video URLs.

Supports two models:

- **SAM3** (default) — Meta's SAM3 via `facebook/sam3` on Hugging Face
- **Falcon Perception** — TII's Falcon Perception via `tiiuae/Falcon-Perception`

## Requirements

- Python 3.13+
- CUDA GPU (recommended for both models)

### SAM3 access

The default model (`facebook/sam3`) is gated. Accept the terms at `https://huggingface.co/facebook/sam3` and authenticate before running:

```bash
hf auth login
```

### Falcon Perception

No authentication is required. The model is downloaded automatically on first use.

## Install

```bash
uv sync
uv run sam --help
```

## Usage

### Image

```bash
uv run sam image -i image.jpg -p "ear"
```

Image URL input also works:

```bash
uv run sam image -i "http://images.cocodataset.org/val2017/000000077595.jpg" -p "ear"
```

Writes an overlay image using the default name `<stem>.sam.png`.

- For local files, the output is written next to the input.
- For URLs, the output is written to the current directory.

Optional output path:

```bash
uv run sam image -i image.jpg -p "ear" -o result.png
```

#### Using Falcon Perception

```bash
uv run sam image -i image.jpg -p "the red car" --model falcon
```

### Video

```bash
uv run sam video -i video.mp4 -p "penguin"
```

Writes `<stem>.sam.mp4` to the current directory. Processing 50 frames by default.

> **Note:** Video segmentation is only available with SAM3 (`--model sam3`, the default). Falcon Perception does not support video yet.

Optional flags:

```bash
# Start from a specific frame
uv run sam video -i video.mp4 -p "penguin" --start-frame 100

# Process more or fewer frames
uv run sam video -i video.mp4 -p "penguin" --max-frames 100

# Get the total frame count (no prompt required)
uv run sam video -i video.mp4 --frames
```

### Segment an entire video in 50-frame chunks

Use the provided `segment.sh` script to process a full video automatically. It queries the frame count, then runs segmentation in 50-frame batches, producing one output file per chunk:

```bash
./segment.sh video.mp4 "penguin"
```

Outputs: `video-1.sam.mp4`, `video-2.sam.mp4`, ...

### Common options

| Flag | Default | Description |
|---|---|---|
| `-i / --input` | required | Path or URL to the input |
| `-p / --prompt` | required | Text prompt to segment |
| `-o / --output` | auto | Path to write the output file |
| `--model` | `sam3` | `sam3` or `falcon` |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--threshold` | `0.5` | Instance confidence threshold |
| `--mask-threshold` | `0.5` | Mask binarisation threshold |

Show all options:

```bash
uv run sam --help
uv run sam image --help
uv run sam video --help
```
