# Direct LLM Baseline Prompt Development

This log documents prompt changes made using development records only. The
independent 300-record test set must not be inspected or used to revise the
direct baseline prompt.

## Data boundary

- Development set: sequence indices 1-50 within each of the four note-length
  strata (200 records total).
- Independent test set: sequence indices 51-125 within each stratum (300
  records total).
- The direct prompt is frozen before any independent-test execution.

## direct-kcd-v1.0

The initial zero-shot prompt requested direct raw-note-to-KCD-8 coding in one
model call. A technical smoke test used two development records per length
stratum (eight records total). After input-format and JSON-parser corrections,
all eight records completed with valid output.

The development-only diagnostic comparison was not treated as a performance
estimate. It showed three general error classes:

1. use of shortened generic ICD-10 codes where KCD-8 required additional
   digits;
2. omission of eligible current diagnoses and symptoms; and
3. unsupported or semantically mismatched diagnosis-code pairs.

## direct-kcd-v1.1

Version 1.1 was revised only in response to those general development error
classes. It:

- explicitly distinguishes KCD-8 from generic WHO ICD-10 and ICD-10-CM;
- adds three KCD-8 format examples rather than patient-note few-shot examples;
- requires consideration of every eligible current diagnosis and symptom;
- strengthens exclusion of imaging-only, historical, vague, and unsupported
  targets;
- requires removal of rule-out qualifiers from output names; and
- adds an internal diagnosis-to-code semantic consistency check.

No curated KCD database, vector search, CLEVER Phase 1 output, or Phase 3 rule
is provided to the direct model. Syntax-only deterministic JSON repair remains
separate from clinical code generation and is recorded in the audit output.

### Development check on eight additional records

Version 1.1 was next run on eight different development records, selected with
seed 20260901 from sequence indices 3-25 (two records per length stratum). All
eight records completed without JSON repair or validation warnings. This check
was used for prompt diagnosis, not as an independent performance estimate.

- Model predictions: 41
- Unique final-reference codes: 37
- Exact-code precision / recall / F1: 0.122 / 0.135 / 0.128
- Three-character main-code precision / recall / F1: 0.632 / 0.686 / 0.658

The large gap between main-code and exact-code agreement showed that the model
often identified the correct disease category but generated unverified extra
digits resembling another ICD variant. Longer notes also continued to produce
unsupported historical or test-derived targets. These are general error
patterns eligible for a subsequent development-only prompt revision; the
individual incorrect code mappings are not to be copied wholesale into the
prompt.

## direct-kcd-v1.2

Version 1.2 addresses the two general error patterns observed in the second
development check. It does not use the third eight-record sample or any
independent-test ground truth.

- It states that KCD-8 code lengths are not uniform and that valid categories
  do not always require a decimal extension.
- It prohibits transferring ICD-10-CM trailing-digit and laterality
  conventions into KCD-8.
- It prohibits appending an unverified `0`, `9`, or decimal place.
- Three short positive/negative format examples illustrate this rule without
  providing patient-note examples.
- It clarifies that long histories, prior tests, procedures, and problem lists
  are contextual evidence rather than automatic coding targets.
- It preserves meaningful clinical modifiers when rule-out wording is removed.

The next check uses a third, non-overlapping eight-record development sample
selected with seed 20260904 from sequence indices 3-25, with two records per
length stratum.

### Development check on a third eight-record sample

Version 1.2 was run once on the prespecified third development sample. All
eight records completed without JSON repair, validation warnings, or failed
records. This development-only check is not an independent performance
estimate.

- Model output items: 39 (37 unique code-by-document predictions)
- Unique final-reference code-by-document targets: 40
- Exact-code TP / FP / FN: 8 / 29 / 32
- Exact-code precision / recall / F1: 0.216 / 0.200 / 0.208
- Three-character main-code TP / FP / FN: 18 / 21 / 22
- Three-character main-code precision / recall / F1: 0.462 / 0.450 / 0.456

Some KCD-8 category and symptom codes were returned in the intended format,
but semantic code errors and unsupported extraction from long notes remained.
One very-long record accounted for substantial over-extraction. Because
versions 1.1 and 1.2 were checked on different non-overlapping eight-record
samples, their diagnostic metrics must not be interpreted as a controlled
head-to-head performance comparison.
