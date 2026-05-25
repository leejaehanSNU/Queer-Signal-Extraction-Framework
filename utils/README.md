# Manual Review App

Streamlit app for reviewing QVI annotations with the associated image and LLM output.

## Run

```bash
streamlit run utils/manual_review_app.py
```

## Default inputs

- Annotations: `outputs/annotations/qvi_annotations.jsonl`
- Fallback annotations: `outputs/annotations/_test_qvi_annotations.jsonl`
- Manifest: `outputs/manifest/gsv_manifest.jsonl`
- Review output: `outputs/reviews/manual_qvi_reviews.jsonl`

## What it saves

Each review is appended to a JSONL file with:

- `verdict` as a boolean
- reviewer note
- image and segment identifiers
- key model fields
- raw model response
