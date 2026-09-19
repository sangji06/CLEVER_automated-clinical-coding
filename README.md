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

The public rule table contains the 20 deterministic rules used in the revised
analysis. It records explicit execution priorities and the controls required
for overlapping rules, pattern-matched component removal, and dual-code output.
The command writes both the corrected Phase 3 output and an applied-rule audit
log. Run the Phase 3 regression tests with:

```bash
python -m unittest discover -s tests -p 'test_phase3_code_corrector.py'
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

Ablation scripts are tracked separately under `scripts/ablation/` and use
local/private inputs and locally built FAISS indexes. The manuscript controls
are:

- Control 1: formal coding-guideline instructions removed from Phase 1 while
  retaining document navigation, extraction engineering, terminology
  refinement, curated KCD retrieval, and Phase 3 correction.
- Control 2: guided Phase 1 + original KCD retrieval.
- Control 3: the Control 1 Phase 1 condition + original KCD retrieval.

The additional direct LLM baseline maps each raw note to final KCD-8 codes in
one Qwen3-32B call, without using any CLEVER component:

```bash
python scripts/baselines/run_direct_llm_kcd.py \
  --input-dir data/private_independent_test_notes \
  --output outputs/direct_llm/direct_kcd_predictions.json \
  --model-id qwen.qwen3-32b-v1:0
```

See `scripts/ablation/README.md` for command examples.
See `scripts/baselines/README.md` for the direct baseline protocol and outputs.

The repository contains synthetic examples for smoke testing and format
validation only. Patient-derived few-shot examples used during the study are
not included. They are not required to inspect the published prompt logic, but
exact regeneration of the original LLM responses requires the corresponding
institutionally governed inputs.

## Annotation App

The Streamlit annotation tool used to support ground-truth construction is included under `tools/annotation_app/`. It is a research utility and is separate from the main CLEVER inference pipeline. Clinical notes and annotation outputs are not included.

## Data

Clinical notes, ground-truth annotations, the curated KCD database, and FAISS
indexes are not included. See `docs/data_availability.md`.

