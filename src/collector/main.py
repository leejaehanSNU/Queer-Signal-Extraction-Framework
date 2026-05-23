from __future__ import annotations

import argparse
import logging
from pathlib import Path

from collector import build_parser, run_collector


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="collector-main")
    parser.add_argument(
        "--segment-file",
        default="/Users/jhmac/Library/Mobile Documents/com~apple~CloudDocs/서울대학교/2026 LQBTQWalkability/QueerSignalExtraction/data/raw/2021_walkbike_with_LGBTQ_density_augmented.csv",
        help="CSV/JSON segment table. Defaults to the project raw CSV with seg_lat and seg_lon columns.",
    )
    parser.add_argument(
        "--provider",
        default="google_street_view",
        help="Image source adapter. Defaults to google_street_view.",
    )
    parser.add_argument(
        "--api-key",
        help="API key override. Defaults to GSV_API_KEY or GOOGLE_STREET_VIEW_API_KEY in .env.",
    )
    parser.add_argument(
        "--headings",
        default="0,90,180,270",
        help="Comma-separated camera headings.",
    )
    parser.add_argument(
        "--pitch",
        type=int,
        default=0,
        help="Camera pitch in degrees.",
    )
    parser.add_argument(
        "--fov",
        type=int,
        default=90,
        help="Field of view in degrees.",
    )
    parser.add_argument(
        "--image-size",
        default="640x640",
        help="Street View image size used by the downloader.",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/jhmac/Library/Mobile Documents/com~apple~CloudDocs/서울대학교/2026 LQBTQWalkability/QueerSignalExtraction/data/raw/images",
        help="Directory where raw images will be written.",
    )
    parser.add_argument(
        "--manifest-path",
        default="/Users/jhmac/Library/Mobile Documents/com~apple~CloudDocs/서울대학교/2026 LQBTQWalkability/QueerSignalExtraction/outputs/manifest/gsv_manifest.jsonl",
        help="Path for the JSONL manifest.",
    )
    parser.add_argument(
        "--captured-at",
        help="Optional capture date string.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level such as INFO or DEBUG.",
    )
    return parser


def main() -> int:
    parser = build_main_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    logger = logging.getLogger("collector.main")
    logger.info("Starting collector")
    try:
        collector_args = build_parser().parse_args(
            [
                "--segment-file",
                args.segment_file,
                "--provider",
                args.provider,
                "--headings",
                args.headings,
                "--pitch",
                str(args.pitch),
                "--fov",
                str(args.fov),
                "--image-size",
                args.image_size,
                "--output-dir",
                args.output_dir,
                "--manifest-path",
                args.manifest_path,
            ]
            + (["--api-key", args.api_key] if args.api_key else [])
            + (["--captured-at", args.captured_at] if args.captured_at else [])
        )
        exit_code = run_collector(collector_args)
        logger.info("Collector finished with exit code %s", exit_code)
        return exit_code
    except Exception as exc:
        logger.exception("Collector failed: %s", exc)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())