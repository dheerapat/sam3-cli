from src.cli import build_parser
from src.image import run_image_command
from src.video import run_video_command


def main():
    args = build_parser().parse_args()
    if args.subcommand == "image":
        run_image_command(args)
    elif args.subcommand == "video":
        run_video_command(args)


if __name__ == "__main__":
    main()
