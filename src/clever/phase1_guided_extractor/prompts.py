"""Prompt templates for CLEVER Phase 1 (Guided Extractor).

The phase prompts below preserve the original research prompt logic from
`entity_extraction_qwen_final.py`, while replacing patient-derived few-shot
examples with synthetic examples and translating field references into English.
"""

STEP1_ENTITY_EXTRACTION_PROMPT = """You are an expert clinical coding assistant. Your task is to extract diagnoses and symptoms for coding, but you MUST first mentally segment the patient's history.
CRITICAL: Calculate all start/end indices based on the SOURCE_TEXT provided, NOT the dictionary format.

**Internal Step A: Mental Segmentation (Do not output this)**
First, locate the text value associated with the key 'Present illness'.
Now, divide ONLY this specific text value into two logical parts:
1. **PMH (Past Medical History):** This is background information. It includes chronic illnesses, lists of past diagnoses, or past procedures found within this text.
2. **PI (Present Illness):** This is the acute story of the current ER visit. It describes recent symptoms and events found within this text.

**Internal Step B: Extraction (This is your JSON output)**
Now, perform the following extraction tasks. Your focus MUST be on information from the **PI** (which you mentally identified in Step A) and other acute sections.

**Extraction Rules:**
1. Label only diagnoses or symptoms directly related to the patient's current hospital visit.
   - This means you should IGNORE most chronic conditions listed in the PMH section.
   - Focus on:
     - Symptoms from the PI block.
     - Extract all diagnoses listed in the 'Clinical diagnosis' and 'Diagnosis' fields.
2. If a diagnosis is marked as "rule out" (e.g., "r/o pneumonia"), treat it as a confirmed diagnosis. Extract ONLY the medical condition itself.
   - Example: If "r/o pneumonia", output "pneumonia". If "r/o d/t AGE", output "AGE". If "r/o d/t HCC", output "HCC".
3. DO NOT LABEL SENTENCES THAT DESCRIBE SYMPTOMS STATED BY THE PATIENT.
   - Example: Exclude phrases such as "the patient reports generalized weakness", "a tearing sensation", or "sudden feeling that the left arm is losing strength".
4. Refer to the 'Review of systems' field to identify additional relevant symptoms.
   - CRITICAL ROS RULE: Only extract the symptom if it is marked as present, which is with a `(+)` sign.
   - DO NOT extract symptoms marked as absent, typically with a `(-)` sign.
   - Example 1: If "Headache/Dizziness (+/-)", you MUST label "Headache" because it is `+`.
   - Example 2: If "cough/sputum/rhinorrhea (+/+/-)", you MUST label "cough" and "sputum" because they are `+`.
   - Example 3: If "general weakness/exertional dyspnea (-/-)", you must NOT label "general weakness" or "exertional dyspnea" because they are `-`.
5. Do not refer to information from the Physical Examination section.
6. Label each diagnosis or symptom individually.
   - Example: If "r/o pneumonia", output "pneumonia". If "r/o d/t AGE", output "AGE". If "r/o d/t HCC", output "HCC".

Your output MUST be ONLY a JSON list of extracted entities, matching the format of the few-shot examples.
"""

STEP2_SUPPORTIVE_EXTRACTION_PROMPT = """For DIAGNOSIS identified, label supplementary information as "Supportive" ONLY IF it falls into the following specific categories. If information does not fit these categories, it MUST be ignored.
CRITICAL: Calculate all start/end indices based on the SOURCE_TEXT provided, NOT the dictionary format.

Label this entire elaborated phrase as "Supportive" if it meets the following criteria:
1. It must contain the full text of the original, shorter DIAGNOSIS.
2. It must add significant clinical detail, such as:
   - Anatomical Site or Specification: e.g., "stomach cancer, fundus", "cellulitis of the left leg".
   - Staging or Severity: e.g., "chronic kidney disease, stage 5", "moderate asthma".
   - Etiology (Cause): e.g., "viral pneumonia".

**Examples (This is the core logic):**
- If DIAGNOSIS is "Stomach cancer" and the text contains "Patient has Stomach cancer, fundus", you MUST label the full phrase "Stomach cancer, fundus" as "Supportive". Do NOT label just "fundus".
- If DIAGNOSIS is "Chronic kidney disease" and the text contains "PMH: Chronic kidney disease, stage 5", you MUST label the full phrase "Chronic kidney disease, stage 5" as "Supportive". Do NOT label just "stage 5".

3. Strict Exclusion Rules:
   - DO NOT LABEL vague descriptions, test results, imaging findings, patient history not directly modifying a current diagnosis, rule-out diagnosis text, or subjective descriptions such as "splitting" or "tearing".
   - For neoplasms, ONLY anatomical site or laterality is supportive. STRICTLY IGNORE morphology, stage, metastasis, or treatment history.
4. If no information matching the allowed categories is found for a primary term, leave it unchanged and output nothing for that term.
"""

STEP3_SELF_VERIFICATION_PROMPT = """Review the list of labeled terms with the medical record and modify or filter out any that are not appropriate according to the rules.
CRITICAL: Calculate all start/end indices based on the SOURCE_TEXT provided, NOT the dictionary format.

Follow these rules:
1. Post-process the entities. If the entities contain the notation "r/o", remove this notation.
2. Exclude past history and chronic diseases that are not directly addressed in the current visit.
3. DO NOT EXCLUDE diagnoses that were extracted from the 'Clinical diagnosis' and 'Diagnosis' fields.
4. Exclude words like "aggravation", "progression", or "origin".
5. Exclude neoplasm information about metastasis, stage, or treatment history.
6. Exclude overly vague disease terms such as just "cancer", "metastasis", or "stone".
7. Exclude findings from imaging reports such as CT or MRI.
8. Exclude clinical findings and laboratory results.
9. Exclude subjective or figurative descriptions of symptoms.
   - Example: Exclude phrases such as "sudden feeling that the left arm is losing strength" or "a severe tearing sensation".
10. If there is no information to exclude, leave it unchanged.
"""

STEP4_ENTITY_REFINEMENT_PROMPT = """Your task is to expand abbreviated terms into their full medical names. This task requires careful consideration of the medical context.

1. Convert all abbreviated terms into their full, formal medical names.
   - Example: 'NF' -> 'Neutropenic fever', 'AGE' -> 'Acute gastroenteritis', 'flu' -> 'Influenza'.
2. Crucially, you MUST refer to the original medical record to expand abbreviations based on the clinical context.
3. Translate any symptoms written in Korean into their English equivalent.
   - Example : '발열' → 'Fever', '열이 난다' → 'Fever', '복통' → 'Abdominal pain', '가슴 답답함' → 'Chest discomfort', '기침' → 'Cough', '구역감' → 'Nausea'.
4. If there is no information to expand or translate, leave it unchanged.
"""

SYNTHETIC_FEW_SHOT_EXAMPLE = """Synthetic example only. This example does not correspond to a real patient record.

Input:
The patient visited the emergency department with fever and cough. Chest imaging suggested pneumonia. A history of hypertension was noted but was not addressed during this visit.

Expected Step 1 output:
[
  {"entity": "fever", "label": "Symptom"},
  {"entity": "cough", "label": "Symptom"},
  {"entity": "pneumonia", "label": "Diagnosis"}
]

Expected Step 4 output:
[
  {"term": "Fever", "original_entities": [{"entity": "fever", "label": "Symptom"}]},
  {"term": "Cough", "original_entities": [{"entity": "cough", "label": "Symptom"}]},
  {"term": "Pneumonia", "original_entities": [{"entity": "pneumonia", "label": "Diagnosis"}]}
]
"""

DEFAULT_PROMPTS = {
    "step1": STEP1_ENTITY_EXTRACTION_PROMPT,
    "step2": STEP2_SUPPORTIVE_EXTRACTION_PROMPT,
    "step3": STEP3_SELF_VERIFICATION_PROMPT,
    "step4": STEP4_ENTITY_REFINEMENT_PROMPT,
}

DEFAULT_FEW_SHOTS = {
    "step1": SYNTHETIC_FEW_SHOT_EXAMPLE,
    "step2": SYNTHETIC_FEW_SHOT_EXAMPLE,
    "step3": SYNTHETIC_FEW_SHOT_EXAMPLE,
    "step4": SYNTHETIC_FEW_SHOT_EXAMPLE,
}
