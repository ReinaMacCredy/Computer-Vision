#!/usr/bin/env python3
"""
Evaluation runner for the CBIR+RAG Image Retrieval System.
Runs retrieval evaluation using ground-truth dataset and calibrates similarity threshold.

Usage:
    uv run python eval/run_eval.py
"""

import os
import sys
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import ImageRAG


def load_ground_truth(path="eval/ground_truth.json"):
    with open(path, "r") as f:
        return json.load(f)


def print_table(title, headers, rows):
    """Print a formatted table to terminal."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"  {header_line}")
    print(f"  {'-+-'.join('-' * w for w in col_widths)}")

    for row in rows:
        row_line = " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(f"  {row_line}")
    print()


def main():
    print(">>> Loading ImageRAG engine...")
    start = time.time()
    engine = ImageRAG()
    print(f"    Engine loaded in {time.time() - start:.1f}s\n")

    gt = load_ground_truth()
    print(f">>> Loaded {len(gt)} evaluation queries from ground_truth.json")

    # -- Retrieval Evaluation --
    results_all = {}
    for k in [5, 10]:
        print(f"\n>>> Running evaluate_retrieval(k={k})...")
        start = time.time()
        summary = engine.evaluate_retrieval(gt, k=k, mode="text_to_image")
        elapsed = time.time() - start
        results_all[f"k={k}"] = summary
        print(f"    Done in {elapsed:.1f}s")

    # Print retrieval results
    headers = ["Metric", "K=5", "K=10"]
    metric_names = [f"Precision@", f"Recall@", f"F1@", f"nDCG@"]
    rows = []
    for m in metric_names:
        k5_key = f"{m}5"
        k10_key = f"{m}10"
        v5 = results_all["k=5"].get(k5_key, 0.0)
        v10 = results_all["k=10"].get(k10_key, 0.0)
        rows.append([k5_key, f"{v5:.4f}", f"{v10:.4f}"])

    mrr5 = results_all["k=5"].get("MRR", 0.0)
    mrr10 = results_all["k=10"].get("MRR", 0.0)
    rows.append(["MRR", f"{mrr5:.4f}", f"{mrr10:.4f}"])

    print_table("Retrieval Evaluation Results", headers, rows)

    # -- Threshold Calibration --
    print(">>> Running threshold calibration...")
    candidate_thresholds = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
    start = time.time()
    cal_result = engine.calibrate_threshold(gt, candidate_thresholds, k=10)
    elapsed = time.time() - start
    print(f"    Done in {elapsed:.1f}s")

    # Print calibration results
    cal_rows = []
    for stat in cal_result["all_stats"]:
        marker = " <-- best" if stat["threshold"] == cal_result["best_threshold"] else ""
        cal_rows.append([
            f"{stat['threshold']:.2f}",
            f"{stat['precision']:.4f}",
            f"{stat['recall']:.4f}",
            f"{stat['f1']:.4f}{marker}"
        ])

    print_table("Threshold Calibration", ["Threshold", "Precision", "Recall", "F1"], cal_rows)
    print(f"  Best threshold: {cal_result['best_threshold']}")
    print(f"  Best F1: {cal_result['best_f1']:.4f}\n")

    # -- Save results --
    output = {
        "retrieval_evaluation": results_all,
        "threshold_calibration": {
            "best_threshold": cal_result["best_threshold"],
            "best_f1": cal_result["best_f1"],
            "all_stats": cal_result["all_stats"]
        }
    }

    output_path = "eval/results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f">>> Results saved to {output_path}")


if __name__ == "__main__":
    main()
