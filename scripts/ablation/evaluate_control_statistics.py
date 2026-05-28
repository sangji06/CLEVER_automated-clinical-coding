"""Statistical comparison for CLEVER ablation/control experiments.

This script preserves the analysis used for the manuscript ablation table:
per-document Exact Match F1 is computed from Phase 3 evaluation workbooks, and
CLEVER is compared against each control with a paired t-test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_evaluation_rows(path: str | Path) -> pd.DataFrame:
    """Load a Phase 3 evaluation workbook and normalize the result column."""
    workbook = pd.ExcelFile(path)
    sheet_name = "Detailed Results" if "Detailed Results" in workbook.sheet_names else workbook.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet_name)

    if "Exact Result" in df.columns and "Result" not in df.columns:
        df = df.rename(columns={"Exact Result": "Result"})

    required_columns = {"Document ID", "Result"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    return df


def compute_per_doc_f1(df: pd.DataFrame) -> dict[str, float]:
    """Compute per-document F1 from TP/FP/FN rows."""
    results: dict[str, float] = {}

    for doc_id, group in df.groupby("Document ID"):
        tp = (group["Result"] == "TP").sum()
        fp = (group["Result"] == "FP").sum()
        fn = (group["Result"] == "FN").sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[str(doc_id)] = f1

    return results


def significance_label(p_value: float) -> str:
    """Return manuscript-style significance stars."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def compare_control(
    clever_arr: np.ndarray,
    control_arr: np.ndarray,
    control_name: str,
) -> dict[str, float | str]:
    """Run paired statistical tests for one control comparison."""
    diff = clever_arr - control_arr
    try:
        from scipy import stats
    except ImportError as error:
        raise ImportError("scipy is required for ablation statistical tests. Install project requirements first.") from error

    t_stat, t_p_value = stats.ttest_rel(clever_arr, control_arr)

    return {
        "Comparison": f"CLEVER vs {control_name}",
        "N common documents": len(clever_arr),
        "CLEVER mean per-document F1": clever_arr.mean(),
        "Control mean per-document F1": control_arr.mean(),
        "Mean F1 difference": diff.mean(),
        "Paired t statistic": t_stat,
        "Paired t-test p-value": t_p_value,
        "Paired t-test significance": significance_label(t_p_value),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired statistical tests for CLEVER ablation/control results."
    )
    parser.add_argument("--clever-eval", required=True, help="CLEVER Phase 3 evaluation XLSX.")
    parser.add_argument("--control1-eval", required=True, help="Control 1 Phase 3 evaluation XLSX.")
    parser.add_argument("--control2-eval", required=True, help="Control 2 Phase 3 evaluation XLSX.")
    parser.add_argument("--control3-eval", required=True, help="Control 3 Phase 3 evaluation XLSX.")
    parser.add_argument("--output-excel", required=True, help="Path to save statistical comparison XLSX.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    clever_f1 = compute_per_doc_f1(load_evaluation_rows(args.clever_eval))
    control_f1 = {
        "Control 1 (w/o guidelines)": compute_per_doc_f1(load_evaluation_rows(args.control1_eval)),
        "Control 2 (w/o curation)": compute_per_doc_f1(load_evaluation_rows(args.control2_eval)),
        "Control 3 (baseline)": compute_per_doc_f1(load_evaluation_rows(args.control3_eval)),
    }

    common_docs = set(clever_f1)
    for values in control_f1.values():
        common_docs &= set(values)
    common_docs = sorted(common_docs)

    if not common_docs:
        raise ValueError("No common Document ID values were found across CLEVER and control workbooks.")

    clever_arr = np.array([clever_f1[doc_id] for doc_id in common_docs])

    summary_rows = []
    per_doc_rows = [{"Document ID": doc_id, "CLEVER F1": clever_f1[doc_id]} for doc_id in common_docs]

    for control_name, values in control_f1.items():
        control_arr = np.array([values[doc_id] for doc_id in common_docs])
        summary_rows.append(compare_control(clever_arr, control_arr, control_name))

        column_name = f"{control_name} F1"
        for row, doc_id in zip(per_doc_rows, common_docs):
            row[column_name] = values[doc_id]
            row[f"CLEVER - {control_name}"] = clever_f1[doc_id] - values[doc_id]

    summary_df = pd.DataFrame(summary_rows)
    per_doc_df = pd.DataFrame(per_doc_rows)

    output_excel = Path(args.output_excel)
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_excel) as writer:
        summary_df.to_excel(writer, sheet_name="Statistical Tests", index=False)
        per_doc_df.to_excel(writer, sheet_name="Per-document F1", index=False)

    print(f"Common documents: {len(common_docs)}")
    print(summary_df.to_string(index=False))
    print(f"Saved ablation statistics to {output_excel}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise



