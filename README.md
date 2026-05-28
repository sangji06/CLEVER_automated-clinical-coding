# CLEVER

CLEVER is an automated clinical coding framework for KCD code assignment from
emergency department clinical notes. The pipeline follows three manuscript
phases:

1. **Phase 1: Guided Extractor** - diagnosis and symptom extraction with
   guideline-informed prompts.
2. **Phase 2: Code Retriever** - KCD code retrieval using SapBERT embeddings and
   a FAISS vector index.
3. **Phase 3: Code Corrector** - rule-based post-processing of retrieved KCD
   codes.

This public repository is organized for code review and reproducibility without
including patient records, institutional clinical assets, the curated KCD
database, or FAISS indexes derived from that database.

## Repository Layout

```text
src/clever/                 Reusable Python package code
scripts/                    Command-line entry points
rules/                      Public post-processing rules
docs/                       Data availability and release notes
examples/                   Synthetic examples only
tools/                      Research utilities, including annotation app
data/                       Placeholder for local/private data
outputs/                    Placeholder for generated outputs
tests/                      Lightweight tests
```

## Basic Usage

Run Phase 1 on local text records:

```bash
python scripts/run_phase1_guided_extractor.py \
  --input-dir data/private_ed_notes \
  --output outputs/phase1_output.json \
  --model-id qwen.qwen3-32b-v1:0
```

Build a FAISS index from a local curated KCD CSV:

```bash
python scripts/build_faiss_index.py \
  --kcd-csv /path/to/curated_kcd.csv \
  --output-index /path/to/faiss_index \
  --device cuda
```

Run Phase 2 only, using an existing Phase 1 JSON output:

```bash
python scripts/run_phase2_code_retriever.py \
  --faiss-index /path/to/faiss_index \
  --phase1-output outputs/phase1_output.json \
  --output-json outputs/phase2_output.json
```

Evaluate saved Phase 2 output cumulatively:

```bash
python scripts/evaluate_pipeline_phase2.py \
  --ground-truth data/private_ground_truth.json \
  --phase2-output outputs/phase2_output.json \
  --output-excel outputs/phase2_evaluation.xlsx
```

Run Phase 3:

```bash
python scripts/run_phase3_code_corrector.py \
  --rules-csv rules/code_correction_rules.csv \
  --input-json outputs/phase2_output.json \
  --output-json outputs/phase3_output.json \
  --log-csv outputs/phase3_applied_rules.csv
```

Evaluate Phase 1 independently:

```bash
python scripts/evaluate_phase1_independent.py \
  --predictions outputs/phase1_output.json \
  --ground-truth data/private_ground_truth.json \
  --output outputs/phase1_evaluation.xlsx
```

Evaluate Phase 2 independently:

```bash
python scripts/evaluate_phase2_independent.py \
  --input-csv data/Step2_code_list.csv \
  --output-excel outputs/phase2_independent_evaluation.xlsx
```

Evaluate cumulative Phase 3 output:

```bash
python scripts/evaluate_pipeline_phase3.py \
  --ground-truth data/private_ground_truth.json \
  --phase3-output outputs/phase3_output.json \
  --output-excel outputs/phase3_evaluation.xlsx
```

Run public Phase 2 -> Phase 3 together, starting from an existing Phase 1 output:

```bash
python scripts/run_pipeline.py \
  --faiss-index /path/to/faiss_index \
  --phase1-output outputs/phase1_output.json \
  --rules-csv rules/code_correction_rules.csv \
  --phase2-output outputs/phase2_output.json \
  --phase3-output outputs/phase3_output.json \
  --rule-log outputs/phase3_applied_rules.csv
```

Ablation scripts are tracked separately under `scripts/ablation/` and use local/private inputs and locally built FAISS indexes. The manuscript controls are:

- Control 1: no-guideline Phase 1 + curated KCD retrieval.
- Control 2: guided Phase 1 + original KCD retrieval.
- Control 3: no-guideline Phase 1 + original KCD retrieval.

See `scripts/ablation/README.md` for command examples.

## Annotation App

The Streamlit annotation tool used to support ground-truth construction is included under `tools/annotation_app/`. It is a research utility and is separate from the main CLEVER inference pipeline. Clinical notes and annotation outputs are not included.
## Data

Clinical notes, ground-truth annotations, the curated KCD database, and FAISS
indexes are not included. See `docs/data_availability.md`.






