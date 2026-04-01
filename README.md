# sam-demo

Minimal CLI for running Meta's SAM3 text-prompted segmentation on a local image or image URL.

## Requirements

- Python 3.13+
- Access to the gated `facebook/sam3` model on Hugging Face

Accept the model terms at `https://huggingface.co/facebook/sam3` and authenticate before running:

```bash
huggingface-cli login
```

## Install

Install the project in editable mode so the `sam` command is available locally:

```bash
uv pip install -e .
```

You can also run it without installing:

```bash
uv run sam --help
```

## Usage

```bash
sam -i image.jpg -p "ear"
```

Image URL input also works:

```bash
sam -i "http://images.cocodataset.org/val2017/000000077595.jpg" -p "ear"
```

This writes an overlay image using the default name `image.sam.png`.

- For local files, the output is written next to the input image.
- For URLs, the output is written to the current directory.

Optional output path:

```bash
sam -i image.jpg -p "ear" -o result.png
```

Show all options:

```bash
sam --help
```
