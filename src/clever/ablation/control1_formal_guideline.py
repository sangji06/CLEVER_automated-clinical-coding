"""Prompt construction for Control 1: formal-guideline ablation.

These prompts are a minimal-deletion copy of the standard Phase 1 prompts.
Only rules classified for removal in the formal-guideline ablation have been
deleted. The four-step structure, study-specific rules, navigation, output
format, and normalization instructions are otherwise preserved.

The executable runner calls :func:`build_control1_prompts` with the original
experiment prompts. Keeping a second, hand-copied Control 1 prompt here would
risk drift, so the experimental prompt exists only as that deterministic
transformation.
"""

from __future__ import annotations

import re


SYNTHETIC_SMOKE_STEP1_FEW_SHOT = """Synthetic format example only. This example does not correspond to a real patient record.

SOURCE_TEXT:
fever; cough; pneumonia

Expected output:
[
  {"entity": "fever", "label": "Symptom"},
  {"entity": "cough", "label": "Symptom"},
  {"entity": "pneumonia", "label": "Diagnosis"}
]
"""


SYNTHETIC_SMOKE_STEP2_FEW_SHOT = """Synthetic format example only. This example does not correspond to a real patient record.

SOURCE_TEXT:
fever; cough; pneumonia, descriptor alpha

PREVIOUS_STEP_RESULTS:
[
  {"entity": "fever", "label": "Symptom"},
  {"entity": "cough", "label": "Symptom"},
  {"entity": "pneumonia", "label": "Diagnosis"}
]

Expected output:
[
  {"entity": "fever", "label": "Symptom"},
  {"entity": "cough", "label": "Symptom"},
  {"entity": "pneumonia", "label": "Diagnosis"}
]
"""


SYNTHETIC_SMOKE_STEP3_FEW_SHOT = """Synthetic format example only. This example does not correspond to a real patient record.

SOURCE_TEXT:
fever; cough; r/o pneumonia aggravation; tearing

PREVIOUS_STEP_RESULTS:
[
  {"entity": "fever", "label": "Symptom"},
  {"entity": "cough", "label": "Symptom"},
  {"entity": "r/o pneumonia aggravation", "label": "Diagnosis"},
  {"entity": "tearing", "label": "Symptom"}
]

Expected output:
[
  {"entity": "fever", "label": "Symptom"},
  {"entity": "cough", "label": "Symptom"},
  {"entity": "pneumonia", "label": "Diagnosis"}
]
"""


SYNTHETIC_SMOKE_STEP4_FEW_SHOT = """Synthetic format example only. This example does not correspond to a real patient record.

PREVIOUS_STEP_RESULTS:
[
  {"entity": "URI", "label": "Diagnosis"}
]

Expected output:
[
  {"term": "Upper respiratory infection", "original_entities": [{"entity": "URI", "label": "Diagnosis"}]}
]
"""


# Public synthetic examples are retained only for isolated parser and smoke
# checks. They are not represented as the patient-derived examples used in the
# matched final experiment, which are not distributed in this repository.
SYNTHETIC_SMOKE_FEW_SHOTS = {
    "step1": SYNTHETIC_SMOKE_STEP1_FEW_SHOT,
    "step2": SYNTHETIC_SMOKE_STEP2_FEW_SHOT,
    "step3": SYNTHETIC_SMOKE_STEP3_FEW_SHOT,
    "step4": SYNTHETIC_SMOKE_STEP4_FEW_SHOT,
}


def _sub_once(
    text: str,
    pattern: str,
    replacement: str,
    *,
    label: str,
    flags: int = 0,
) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise ValueError(f"Could not apply the expected Control 1 deletion: {label}")
    return updated


def build_control1_prompts(full_prompts: dict[str, str]) -> dict[str, str]:
    """Delete formal-guideline content from the actual experiment prompts.

    The transformation starts from the supplied original strings so retained
    field names, examples, emphasis, spelling, and formatting stay identical.
    Each edit must match exactly once; otherwise the run stops before loading
    any evaluation records.
    """
    missing = {"step1", "step2", "step3", "step4"} - set(full_prompts)
    if missing:
        raise ValueError(f"Missing original prompt steps: {sorted(missing)}")

    step1 = full_prompts["step1"]
    step1 = _sub_once(
        step1,
        r", but you MUST first mentally segment the patient's history\.",
        ".",
        label="Step 1 segmentation task framing",
    )
    step1 = _sub_once(
        step1,
        r"\n\s+\*\*Internal Step A: Mental Segmentation \(Do not output this\)\*\*.*?(?=\n\s+\*\*Internal Step B: Extraction)",
        (
            "\n    **Internal Step A: Document Navigation (Do not output this)**\n"
            "    First, locate the text value associated with the key "
            "'현병력-Free Text #13'.\n    "
        ),
        label="Step 1 PMH/PI segmentation block",
        flags=re.DOTALL,
    )
    step1 = _sub_once(
        step1,
        r"Now, perform the following extraction tasks\. Your focus MUST be on information from the \*\*PI\*\* \(which you mentally identified in Step A\) and other acute sections\.",
        "Now, perform the following extraction tasks.",
        label="Step 1 acute-section focus",
    )
    step1 = _sub_once(
        step1,
        r"\n\s+1\.\s+Label only diagnoses or symptoms.*?(?=\n\s+2\.\s+If a diagnosis)",
        (
            "\n    1.  Extract all diagnoses listed in the "
            "'임상진단명 자동완성검색기능#29' and '진단명-Free Text#30' fields."
        ),
        label="Step 1 current-visit and PMH selection rule",
        flags=re.DOTALL,
    )

    # Step 2 is entirely the guideline-based specificity operation. Keep its
    # pipeline slot and JSON interface, but neutralize its behavior as an exact
    # pass-through. The matched runner implements this copy deterministically
    # rather than asking the LLM to reproduce the list.
    step2 = """This is a pass-through control step.
Return PREVIOUS_STEP_RESULTS exactly as received.
Do not add, remove, relabel, merge, split, or rewrite any entity.
Your output MUST be ONLY the same JSON list, with no prose or markdown fences."""

    step3 = full_prompts["step3"]
    step3 = _sub_once(
        step3,
        r"\n\s+2\. Exclude past history and chronic diseases that are not directly addressed in the current visit\.",
        "",
        label="Step 3 past-history and chronic-disease exclusion",
    )
    control1_prompts = {
        "step1": step1,
        "step2": step2,
        "step3": step3,
        "step4": full_prompts["step4"],
    }

    required_retained_text = {
        "step1 navigation": (
            "step1",
            "First, locate the text value associated with the key "
            "'현병력-Free Text #13'.",
        ),
        "step1 diagnosis fields": (
            "step1",
            "'임상진단명 자동완성검색기능#29' and '진단명-Free Text#30'",
        ),
        "step1 ROS notation": ("step1", "CRITICAL ROS RULE"),
        "step3 vague-term normalization": ("step3", "Exclude too vague disease"),
        "step3 imaging exclusion": ("step3", "Exclude findings from the imaging report"),
        "step3 laboratory exclusion": ("step3", "Exclude clinical findings and laboratory results"),
    }
    missing_retained = [
        label
        for label, (step, text) in required_retained_text.items()
        if text not in control1_prompts[step]
    ]
    if missing_retained:
        raise ValueError(
            "Control 1 unexpectedly removed retained navigation/ED-specific content: "
            + ", ".join(missing_retained)
        )

    forbidden_formal_text = {
        "PMH/PI segmentation": ("step1", "PMH (Past Medical History)"),
        "current-visit selection": (
            "step1",
            "Label only diagnoses or symptoms *directly related to the patient's current hospital visit*",
        ),
        "Step 3 history exclusion": (
            "step3",
            "Exclude past history and chronic diseases that are not directly addressed in the current visit",
        ),
    }
    retained_formal = [
        label
        for label, (step, text) in forbidden_formal_text.items()
        if text in control1_prompts[step]
    ]
    if retained_formal:
        raise ValueError(
            "Control 1 still contains ablated formal-guideline content: "
            + ", ".join(retained_formal)
        )

    return control1_prompts
