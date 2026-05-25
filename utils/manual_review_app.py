from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = ROOT / "outputs" / "annotations" / "qvi_annotations.jsonl"
DEFAULT_TEST_ANNOTATIONS = ROOT / "outputs" / "annotations" / "_test_qvi_annotations.jsonl"
DEFAULT_MANIFEST = ROOT / "outputs" / "manifest" / "gsv_manifest.jsonl"
DEFAULT_REVIEW_LOG = ROOT / "outputs" / "reviews" / "manual_qvi_reviews.jsonl"


def pick_default_annotation_path() -> Path:
    if DEFAULT_ANNOTATIONS.exists():
        return DEFAULT_ANNOTATIONS
    if DEFAULT_TEST_ANNOTATIONS.exists():
        return DEFAULT_TEST_ANNOTATIONS
    return DEFAULT_ANNOTATIONS


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_manifest_index(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        image_id = str(record.get("image_id", ""))
        if image_id:
            index[image_id] = record
    return index


def load_review_index(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        image_id = str(record.get("image_id", ""))
        if image_id:
            latest[image_id] = record
    return latest


def resolve_image_path(annotation: dict[str, Any], manifest_record: dict[str, Any] | None) -> Path | None:
    candidate_paths = [
        annotation.get("file_path"),
        annotation.get("image_path"),
        (manifest_record or {}).get("file_path"),
    ]
    for raw_path in candidate_paths:
        if not raw_path:
            continue
        candidate = Path(str(raw_path)).expanduser()
        if candidate.exists():
            return candidate
        if not candidate.is_absolute():
            relative_candidate = ROOT / candidate
            if relative_candidate.exists():
                return relative_candidate
    return None


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def parse_model_response(raw_value: Any) -> dict[str, Any] | None:
    if isinstance(raw_value, dict):
        return raw_value
    if not raw_value:
        return None
    if not isinstance(raw_value, str):
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return None


def review_payload(annotation: dict[str, Any], verdict: bool, reviewer_note: str) -> dict[str, Any]:
    return {
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "reviewer_note": reviewer_note.strip() or None,
        "image_id": annotation.get("image_id"),
        "segment_id": annotation.get("segment_id"),
        "scene_id": annotation.get("scene_id"),
        "source": annotation.get("source"),
        "file_path": annotation.get("file_path"),
        "environment_type": annotation.get("environment_type"),
        "overall_queer_visibility_score": annotation.get("overall_queer_visibility_score"),
        "overall_confidence": annotation.get("overall_confidence"),
        "rainbow_signal_present": annotation.get("rainbow_signal_present"),
        "inclusive_signage_present": annotation.get("inclusive_signage_present"),
        "rationale_short": annotation.get("rationale_short"),
        "raw_model_response": annotation.get("raw_model_response"),
    }


def append_review(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def next_unreviewed_index(items: list[dict[str, Any]], review_index: dict[str, dict[str, Any]]) -> int:
    for position, item in enumerate(items):
        if str(item.get("image_id", "")) not in review_index:
            return position
    return 0


def initialize_state(items: list[dict[str, Any]], review_index: dict[str, dict[str, Any]]) -> None:
    if "current_index" not in st.session_state:
        st.session_state.current_index = next_unreviewed_index(items, review_index)
    if "verdict" not in st.session_state:
        st.session_state.verdict = True
    if "reviewer_note" not in st.session_state:
        st.session_state.reviewer_note = ""


def move_index(delta: int, items: list[dict[str, Any]]) -> None:
    if not items:
        st.session_state.current_index = 0
        return
    st.session_state.current_index = (st.session_state.current_index + delta) % len(items)

def jump_to_next_unreviewed(items: list[dict[str, Any]], review_index: dict[str, dict[str, Any]]) -> None:
    st.session_state.current_index = next_unreviewed_index(items, review_index)

def advance_to_next_item(current_index: int, items: list[dict[str, Any]]) -> int:
    if not items:
        return 0
    return (current_index + 1) % len(items)

def main() -> None:
    st.set_page_config(page_title="QVI Manual Review", layout="wide")
    st.title("QVI 수동 검수 GUI")
    st.caption("이미지와 LLM 결과를 확인하고 boolean verdict를 저장합니다.")

    annotation_path = Path(
        st.sidebar.text_input("Annotations JSONL", value=str(pick_default_annotation_path()))
    )
    manifest_path = Path(
        st.sidebar.text_input("Manifest JSONL", value=str(DEFAULT_MANIFEST))
    )
    review_path = Path(
        st.sidebar.text_input("Review output JSONL", value=str(DEFAULT_REVIEW_LOG))
    )
    only_unreviewed = st.sidebar.checkbox("Show only unreviewed", value=False)
    rainbow_only = st.sidebar.checkbox("Show only rainbow_signal_present", value=False)
    symbolic_density_over_ = st.sidebar.checkbox("Show only items with symbolic_density >= 1", value=False)
    ped_activity_over_ = st.sidebar.checkbox("Show only items with pedestrian_activity_level >= 3", value=False)
    st.sidebar.divider()
    st.sidebar.write(f"Annotations: {annotation_path}")
    st.sidebar.write(f"Manifest: {manifest_path}")
    st.sidebar.write(f"Reviews: {review_path}")

    annotations = load_jsonl(annotation_path)
    manifest_index = load_manifest_index(manifest_path)
    review_index = load_review_index(review_path)
    initialize_state(annotations, review_index)

    if only_unreviewed:
        visible_items = [item for item in annotations if str(item.get("image_id", "")) not in review_index]
    else:
        visible_items = annotations

    if rainbow_only:
        visible_items = [item for item in visible_items if bool(item.get("rainbow_signal_present", False))]

    if symbolic_density_over_:
        visible_items = [
            item for item in visible_items if item["symbolic_density_score"] >= 1
        ]

    if ped_activity_over_:
        visible_items = [
            item for item in visible_items if item.get("pedestrian_activity_level", 0) >= 3
        ]

    if not annotations:
        st.error("검수할 annotation 파일을 찾지 못했습니다.")
        st.stop()

    if not visible_items:
        st.info("조건에 맞는 항목이 없습니다.")
        st.stop()

    st.session_state.current_index = min(st.session_state.current_index, len(visible_items) - 1)
    current_item = visible_items[st.session_state.current_index]
    current_image_id = str(current_item.get("image_id", ""))
    current_review = review_index.get(current_image_id)
    current_manifest = manifest_index.get(current_image_id)
    image_path = resolve_image_path(current_item, current_manifest)
    parsed_model_response = parse_model_response(current_item.get("raw_model_response"))

    total_count = len(visible_items)
    reviewed_count = sum(1 for item in visible_items if str(item.get("image_id", "")) in review_index)
    st.subheader(f"{st.session_state.current_index + 1} / {total_count}")
    st.progress(reviewed_count / total_count if total_count else 0.0)

    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    with left_col:
        st.markdown("### Image")
        if image_path and image_path.exists():
            st.image(str(image_path), use_container_width=True, caption=str(image_path))
        else:
            st.warning("이미지 경로를 찾지 못했습니다.")
            st.code(safe_json({
                "annotation_file_path": current_item.get("file_path"),
                "manifest_file_path": (current_manifest or {}).get("file_path"),
                "resolved_image_path": str(image_path) if image_path else None,
            }))

        nav_prev_col, skip_col = st.columns(2, gap="small")
        with nav_prev_col:
            prev_clicked = st.button("Prev", use_container_width=True)
        with skip_col:
            skip_clicked = st.button("Skip", use_container_width=True)

        if prev_clicked:
            move_index(-1, visible_items)
            st.rerun()
        if skip_clicked:
            st.session_state.current_index = advance_to_next_item(st.session_state.current_index, visible_items)
            st.rerun()

        verdict_default = bool(current_review.get("verdict", True)) if current_review else True
        st.session_state.verdict = st.radio(
            "Boolean verdict",
            options=[True, False],
            format_func=lambda value: "True / accept" if value else "False / reject",
            index=0 if verdict_default else 1,
            horizontal=True,
        )
        st.session_state.reviewer_note = st.text_area(
            "Reviewer note",
            value=st.session_state.reviewer_note,
            placeholder="Optional: why this annotation should be accepted or rejected.",
            height=120,
        )

        st.markdown("#### Review actions")
        save_col, next_col, next_unreviewed_clicked = st.columns(3, gap="small")
        with save_col:
            save_clicked = st.button("Save decision", type="primary", use_container_width=True)
        with next_col:
            next_clicked = st.button("Save & next", use_container_width=True)
        with next_unreviewed_clicked:
            next_unreviewed_clicked = st.button("Next unreviewed", use_container_width=True)

        if next_unreviewed_clicked:
            jump_to_next_unreviewed(visible_items, review_index)
            st.rerun()
        if save_clicked or next_clicked:
            payload = review_payload(current_item, bool(st.session_state.verdict), st.session_state.reviewer_note)
            append_review(review_path, payload)
            st.success("Saved review decision.")
            review_index = load_review_index(review_path)
            if next_clicked:
                st.session_state.current_index = advance_to_next_item(st.session_state.current_index, visible_items)
            st.rerun()

    with right_col:
        st.markdown("### LLM result")
        # meta_cols = st.columns(2)
        # meta_cols[0].metric("segment_id", str(current_item.get("segment_id", "")))
        # meta_cols[1].metric("image_id", current_image_id)
        # metric_cols = st.columns(3)
        # metric_cols[0].metric("QVI score", str(current_item.get("overall_queer_visibility_score", "")))
        # metric_cols[1].metric("confidence", str(current_item.get("overall_confidence", "")))
        # metric_cols[2].metric("environment", str(current_item.get("environment_type", "")))

        st.markdown("#### Rationale")
        st.write(str(current_item.get("rationale_short", "")))

        st.markdown("#### Structured annotation")
        st.json({key: value for key, value in current_item.items() if key != "raw_model_response"})

        st.markdown("#### Raw model response")
        if parsed_model_response is not None:
            st.json(parsed_model_response)
        else:
            st.code(str(current_item.get("raw_model_response", "")), language="json")

        if current_review:
            st.markdown("#### Existing review")
            st.json(current_review)

    st.caption(
        f"Loaded {len(annotations)} annotations | {len(review_index)} reviewed | output: {review_path}"
    )


if __name__ == "__main__":
    main()
