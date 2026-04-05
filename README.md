# SAM3-CLI 

![](/showcase.png)

Minimal CLI for running Meta's SAM3 text-prompted segmentation on local images, image URLs, local videos, and video URLs.

## Requirements

- Python 3.13+
- Access to the gated `facebook/sam3` model on Hugging Face

Accept the model terms at `https://huggingface.co/facebook/sam3` and authenticate before running:

```bash
huggingface-cli login
```

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

### Video

```bash
uv run sam video -i video.mp4 -p "penguin"
```

Writes `<stem>.sam.mp4` to the current directory. Processing 50 frames by default.

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
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--threshold` | `0.5` | Instance confidence threshold |
| `--mask-threshold` | `0.5` | Mask binarisation threshold |

Show all options:

```bash
uv run sam --help
uv run sam image --help
uv run sam video --help
```
