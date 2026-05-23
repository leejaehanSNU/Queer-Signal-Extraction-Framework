from __future__ import annotations

import argparse
import logging
from pathlib import Path

from annotator import build_parser, run_annotator


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="annotator-main")
    parser.add_argument(
        "--image-dir",
        default="/Users/jhmac/Library/Mobile Documents/com~apple~CloudDocs/서울대학교/2026 LQBTQWalkability/QueerSignalExtraction/data/raw/images",
        help="Directory containing the collected street imagery.",
    )
    parser.add_argument(
        "--manifest-path",
        default="/Users/jhmac/Library/Mobile Documents/com~apple~CloudDocs/서울대학교/2026 LQBTQWalkability/QueerSignalExtraction/outputs/manifest/gsv_manifest.jsonl",
        help="JSONL manifest produced by the collector stage.",
    )
    parser.add_argument(
        "--prompt-template-path",
        default="/Users/jhmac/Library/Mobile Documents/com~apple~CloudDocs/서울대학교/2026 LQBTQWalkability/QueerSignalExtraction/prompts/annotation_prompt.json",
        help="Prompt template JSON for QSEF annotation.",
    )
    parser.add_argument(
        "--schema-path",
        default="/Users/jhmac/Library/Mobile Documents/com~apple~CloudDocs/서울대학교/2026 LQBTQWalkability/QueerSignalExtraction/schemas/qvi_annotation.schema.json",
        help="JSON schema used to validate Gemini output.",
    )
    parser.add_argument(
        "--model-name",
        default="gemini-2.5-flash",
        help="Gemini model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images to process per batch.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Gemini generation temperature.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/Users/jhmac/Library/Mobile Documents/com~apple~CloudDocs/서울대학교/2026 LQBTQWalkability/QueerSignalExtraction/outputs/annotations/qvi_annotations.jsonl",
        help="Path for raw image-level QVI annotations.",
    )
    parser.add_argument(
        "--output-table",
        default="/Users/jhmac/Library/Mobile Documents/com~apple~CloudDocs/서울대학교/2026 LQBTQWalkability/QueerSignalExtraction/outputs/qvi_raw/qvi_summary.csv",
        help="Path for the segment-level summary table.",
    )
    parser.add_argument(
        "--api-key",
        help="Gemini API key override. Defaults to GEMINI_API_KEY, GOOGLE_GENAI_API_KEY, GOOGLE_API_KEY, or GOOGLE_STREET_VIEW_API_KEY in .env.",
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
    logger = logging.getLogger("annotator.main")
    logger.info("Starting annotator")
    try:
        annotator_args = build_parser().parse_args(
            [
                "--image-dir",
                args.image_dir,
                "--manifest-path",
                args.manifest_path,
                "--prompt-template-path",
                args.prompt_template_path,
                "--schema-path",
                args.schema_path,
                "--model-name",
                args.model_name,
                "--batch-size",
                str(args.batch_size),
                "--temperature",
                str(args.temperature),
                "--output-jsonl",
                args.output_jsonl,
                "--output-table",
                args.output_table,
            ]
            + (["--api-key", args.api_key] if args.api_key else [])
        )
        exit_code = run_annotator(annotator_args)
        logger.info("Annotator finished with exit code %s", exit_code)
        return exit_code
    except Exception as exc:
        logger.exception("Annotator failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())