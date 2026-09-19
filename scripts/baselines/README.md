# Direct LLM Baseline

The direct baseline sends each raw emergency-department note to Qwen3-32B once
and asks for final KCD-8 codes. It is independent of CLEVER's Phase 1 output,
curated KCD vector database, SapBERT retrieval, and Phase 3 correction rules.

Freeze the prompt in `src/clever/direct_baseline/prompts.py` using development
data only. Do not edit that prompt after opening the independent test set.

```bash
python scripts/baselines/run_direct_llm_kcd.py \
  --input-dir data/private_independent_test_notes \
  --output outputs/direct_llm/direct_kcd_predictions.json \
  --model-id qwen.qwen3-32b-v1:0 \
  --region us-west-2 \
  --max-workers 1 \
  --batch-size 50
```

The runner writes three final files:

- `direct_kcd_predictions.json`: evaluation-compatible document-to-code lists.
- `direct_kcd_predictions_audit.json`: parsing status, warnings, and response hashes.
- `direct_kcd_predictions_manifest.json`: prompt hash, model, inference settings,
  record counts, and the single-run declaration.

Batch files are kept in a separate directory so interrupted runs can resume.
An existing batch directory can only be resumed when the prompt hash, model,
input directory, and inference settings are identical.

The parser also records `json_repair_applied` and `json_repairs` when it makes
the narrowly defined deterministic repair of inserting a missing object opener
before a top-level JSON key. Diagnosis and code values are never inferred or
rewritten by this repair.

The run manifest also records `input_formatter_version`. The input formatter
supports both valid Python-like note dictionaries and the quoted multiline
pseudo-literal export used by the private note files. In both cases, it sends
the same labeled note fields to the model without using any CLEVER output.

Evaluate the final predictions with the same cumulative code evaluator used for
CLEVER's final output:

```bash
python scripts/evaluate_pipeline_phase3.py \
  --ground-truth data/private_ground_truth.json \
  --phase3-output outputs/direct_llm/direct_kcd_predictions.json \
  --output-excel outputs/direct_llm/direct_kcd_evaluation.xlsx
```
