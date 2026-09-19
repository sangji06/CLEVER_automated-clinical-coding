"""Deterministic Phase 3 code-correction engine for CLEVER."""

from __future__ import annotations

import copy
import csv
import json
import re
from pathlib import Path


def _split_csv_list(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _as_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


class KCDRuleProcessor:
    """Load and apply the versioned Phase 3 rule table.

    The rule table defines explicit execution priority and narrow controls for
    pattern-matched component removal and replacement of pre-existing output
    codes. Removals are limited to the rows matched by a rule.
    """

    def __init__(self) -> None:
        self.rules: list[dict] = []

    def load_rules(self, csv_file_path: str | Path) -> bool:
        """Load rules from CSV and order them by priority."""
        try:
            with Path(csv_file_path).open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            rules = []
            for row_index, row in enumerate(rows):
                exact_desc_raw = str(row.get("exact_descriptions") or "").split(",")
                rule = {
                    "rule_id": int(row["rule_id"]),
                    "exact_codes": _split_csv_list(row.get("exact_codes")),
                    "exact_descriptions": [part.strip() for part in exact_desc_raw],
                    "pattern_codes": str(row.get("pattern_codes") or "").strip(),
                    "pattern_descriptions": str(row.get("pattern_descriptions") or "").strip(),
                    "action_type": str(row.get("action_type") or "").strip(),
                    "remove_codes": _split_csv_list(row.get("remove_codes")),
                    "add_codes": _split_csv_list(row.get("add_code")),
                    "add_diagnoses": _split_csv_list(row.get("add_diagnosis")),
                    "description": str(row.get("description") or "").strip(),
                    "remove_pattern_match": _as_bool(row.get("remove_pattern_match")),
                    "replace_existing_add_codes": _as_bool(row.get("replace_existing_add_codes")),
                    "priority": int(row.get("priority") or (1000 + row_index)),
                    "row_index": row_index,
                }
                rule["pattern_regex"] = (
                    re.compile(rule["pattern_codes"]) if rule["pattern_codes"] else None
                )
                rules.append(rule)

            self.rules = sorted(rules, key=lambda rule: (rule["priority"], rule["row_index"]))
            print(f"Loaded {len(self.rules)} rules.")
            return True
        except Exception as exc:
            print(f"Failed to load rule file: {exc}")
            self.rules = []
            return False

    @staticmethod
    def check_description_match(target_desc: object, condition_desc: str) -> bool:
        if not condition_desc:
            return True
        target = str(target_desc or "").lower()
        for condition in condition_desc.lower().split("|"):
            condition = condition.strip()
            if not condition:
                continue
            if condition.startswith("*") and condition.endswith("*"):
                if condition[1:-1] in target:
                    return True
            elif target == condition:
                return True
        return False

    def apply_rules_to_record(self, record_id: str, record_items: list[dict]):
        """Apply every eligible rule to one model output record."""
        current_items = copy.deepcopy(record_items)
        applied_logs = []

        for rule in self.rules:
            exact_indices: list[int] = []
            used_indices: set[int] = set()
            exact_success = True

            for position, required_code in enumerate(rule["exact_codes"]):
                condition = (
                    rule["exact_descriptions"][position]
                    if position < len(rule["exact_descriptions"])
                    else ""
                )
                found = None
                for index, item in enumerate(current_items):
                    if index in used_indices:
                        continue
                    if (
                        item.get("kcd_code") == required_code
                        and self.check_description_match(
                            item.get("kcd_description", ""), condition
                        )
                    ):
                        found = index
                        break
                if found is None:
                    exact_success = False
                    break
                used_indices.add(found)
                exact_indices.append(found)

            if not exact_success:
                continue

            pattern_index = None
            if rule["pattern_regex"]:
                for index, item in enumerate(current_items):
                    if index in used_indices:
                        continue
                    if (
                        rule["pattern_regex"].match(str(item.get("kcd_code") or ""))
                        and self.check_description_match(
                            item.get("kcd_description", ""),
                            rule["pattern_descriptions"],
                        )
                    ):
                        pattern_index = index
                        break
                if pattern_index is None:
                    continue

            remove_indices = {
                index
                for index in exact_indices
                if current_items[index].get("kcd_code") in rule["remove_codes"]
            }
            if rule["remove_pattern_match"] and pattern_index is not None:
                remove_indices.add(pattern_index)
            if rule["replace_existing_add_codes"]:
                remove_indices.update(
                    index
                    for index, item in enumerate(current_items)
                    if item.get("kcd_code") in rule["add_codes"]
                )

            removed_log = [
                f"{item.get('kcd_code')}({item.get('term', 'NO_TERM')})"
                for index, item in enumerate(current_items)
                if index in remove_indices
            ]
            current_items = [
                item
                for index, item in enumerate(current_items)
                if index not in remove_indices
            ]

            added_log = []
            for position, new_code in enumerate(rule["add_codes"]):
                diagnosis = (
                    rule["add_diagnoses"][position]
                    if position < len(rule["add_diagnoses"])
                    else "Generated Diagnosis"
                )
                current_items.append(
                    {
                        "term": diagnosis,
                        "kcd_code": new_code,
                        "kcd_description": diagnosis,
                    }
                )
                added_log.append(f"{new_code}({diagnosis})")

            applied_logs.append(
                {
                    "record_id": record_id,
                    "rule_id": rule["rule_id"],
                    "priority": rule["priority"],
                    "description": rule["description"],
                    "removed": ", ".join(removed_log),
                    "added": ", ".join(added_log),
                }
            )

        return current_items, applied_logs

    def process_json(
        self,
        input_json_path: str | Path,
        output_json_path: str | Path,
        log_csv_path: str | Path,
    ) -> None:
        """Apply loaded rules to all records and save outputs and an audit log."""
        with Path(input_json_path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        processed_data = {}
        all_logs = []
        for record_id, items in data.items():
            modified_items, logs = self.apply_rules_to_record(record_id, items)
            processed_data[record_id] = modified_items
            all_logs.extend(logs)

        output_path = Path(output_json_path)
        log_path = Path(log_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(processed_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        fieldnames = ["record_id", "rule_id", "priority", "description", "removed", "added"]
        with log_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_logs)

        print(f"Applied {len(all_logs)} rules.")
        print(f"Output file: {output_path}")
        print(f"Log file: {log_path}")
