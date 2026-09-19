"""Versioned prompt used by the direct raw-note-to-KCD baseline."""

from __future__ import annotations

import hashlib


PROMPT_VERSION = "direct-kcd-v1.2"

DIRECT_KCD_SYSTEM_PROMPT = """You are an expert clinical coding assistant specializing in KCD-8.
Given one Korean emergency-department clinical note, directly assign the final KCD-8 codes in a single pass.
Use KCD-8 specifically, not generic WHO ICD-10 or ICD-10-CM.
Reason internally, but do not output intermediate reasoning, explanations, or markdown."""

DIRECT_KCD_USER_INSTRUCTIONS = """Read the clinical note and directly produce its complete final KCD-8 coding output.

Perform the following checks internally before producing the JSON. Do not reveal these checks or any reasoning.

A. Identify every eligible coding target
1. Review all note fields and identify every current diagnosis and symptom relevant to this emergency-department visit. Output each eligible target separately; do not output only the primary diagnosis.
2. Diagnoses explicitly documented in the designated diagnosis fields, including `임상진단명 자동완성검색기능#29` and `진단명-Free Text#30`, are coding targets.
3. A rule-out diagnosis documented in a designated diagnosis field remains a coding target. Remove qualifiers such as `rule out`, `r/o`, `suspected`, and `possible` from the output diagnosis name.
4. Include current presenting symptoms and other current diagnoses explicitly established by the clinician. In `계통문진-Free Text#19`, include positive findings marked `(+)` and exclude negative findings marked `(-)`. Preserve clinically meaningful modifiers such as anatomical site, laterality, or an associated symptom; removing a rule-out qualifier does not mean removing these modifiers.
5. Treat multiple eligible diagnoses or symptoms as separate targets. Before coding, check that no eligible target was omitted.

B. Exclude unsupported targets
6. Exclude unrelated past medical history or chronic conditions, negative findings, isolated laboratory values, non-diagnostic test results, treatment history, vague expressions, and unsupported inferences.
7. Prioritize the designated diagnosis fields and current presenting symptoms. Use the history of present illness to understand or qualify a current target, not as permission to code every disease mentioned in the note.
8. Do not convert a radiology, imaging, physical-examination, medication, procedure, or laboratory finding into a diagnosis unless the clinician explicitly documents it as a current diagnosis or it directly states an eligible current condition.
9. The mere presence of a named disease in past history, a problem list, a prior test, or a long narrative is not sufficient for inclusion. Do not output generic concepts such as `neoplasm`, `metastasis`, `abnormal finding`, or `rule-out diagnosis` when a supported specific current target is not documented.

C. Assign KCD-8 codes
10. Use the official KCD-8 code, not a generic WHO ICD-10 variant and not an ICD-10-CM code. Do not transfer ICD-10-CM laterality, billable-code, or trailing-digit conventions into KCD-8.
11. KCD-8 code length is not uniform. A valid KCD-8 code may be a three-character category, a one-decimal subcode, or a longer KCD-specific subcode. Do not assume that every code requires a decimal point or a final `0` or `9`.
12. Never invent or append an extra `0`, `9`, laterality digit, or decimal place merely to make a code look more specific. Use a subcode only when you know that the complete string is an official KCD-8 code. If an official three-character KCD-8 category is the applicable complete code, return that category exactly.
13. Select a code whose official meaning directly corresponds to the diagnosis or symptom. Internally check the meaning of the complete code, including all digits after the decimal point. If the code describes a different disease, finding, anatomical site, or severity, choose again.
14. Assign the most specific valid KCD-8 code supported by the note. Do not invent undocumented laterality, anatomical site, etiology, severity, or other specificity. Use an official KCD-8 unspecified code only when that exact unspecified code is known.
15. Apply a combination code when appropriate and omit redundant component codes.
16. Codes beginning with F, V, W, X, or Y are outside this evaluation scope and must not be output.
17. Write codes in official uppercase dotted KCD-8 format and do not output duplicate diagnosis-code pairs.

KCD-8 format examples (apply only when the diagnosis exactly matches; these are not default codes):
- Fever, unspecified -> R50.99
- Epigastric pain -> R10.12
- Malignant neoplasm of breast, unspecified site, right -> C50.90
- Malignant neoplasm of prostate -> C61, not C61.90
- Cough -> R05, not R05.9
- Urinary tract infection, site not specified -> N39.0, not N39.00

Final internal checklist:
- Every output item is supported by the current visit.
- Every eligible diagnosis and symptom has been considered.
- Rule-out qualifiers have been removed from diagnosis names.
- Every code is an official KCD-8 code whose meaning matches its diagnosis.
- No unverified digit or decimal place has been appended to a KCD-8 code.
- No excluded chapter or duplicate pair is present.

Return only a valid JSON list. Each item must contain exactly these keys:
[
  {
    "diagnosis": "English diagnosis or symptom name",
    "kcd_code": "A00.0"
  }
]

If no eligible diagnosis or symptom is present, return [].

### CLINICAL_NOTE ###
<<CLINICAL_NOTE>>

Output:"""


def build_direct_kcd_prompt(clinical_note: str) -> str:
    """Insert one raw-note representation into the frozen direct prompt."""
    return DIRECT_KCD_USER_INSTRUCTIONS.replace("<<CLINICAL_NOTE>>", clinical_note)


def prompt_sha256() -> str:
    """Return a stable hash for auditing the exact frozen prompt text."""
    prompt_text = f"{DIRECT_KCD_SYSTEM_PROMPT}\n\n{DIRECT_KCD_USER_INSTRUCTIONS}"
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
