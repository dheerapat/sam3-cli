import argparse
from pathlib import Path


def _add_common_args(parser):
    """Add arguments shared by both image and video subcommands."""
    parser.add_argument("-i", "--input", required=True, help="Path or URL to the input")
    parser.add_argument("-p", "--prompt", help="Text prompt to segment")
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
        "--model",
        choices=("sam3", "falcon"),
        default="sam3",
        help="Segmentation model to use (default: sam3)",
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
    video_parser.add_argument(
        "--frames",
        action="store_true",
        help="Print the total number of frames in the video and exit",
    )

    return parser
