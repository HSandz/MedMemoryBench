import argparse
import json
import os
from collections import defaultdict


def load_all_rag_results(output_dir: str):
    """
    Load all rag_results*.json files under output_dir and return aggregated samples.
    Returns (samples, overall_metrics_files) where samples is a list of per-question dicts.
    """
    samples = []
    overall_metrics_files = []

    def _collect_from_obj(obj):
        """Recursively collect QuerySolution-like dicts from arbitrary JSON shapes."""
        if isinstance(obj, list):
            for item in obj:
                _collect_from_obj(item)
            return
        if not isinstance(obj, dict):
            return
        # Direct QuerySolution dict
        if "question" in obj and ("sample_metrics" in obj or "metrics" in obj):
            # normalize field name
            if "sample_metrics" not in obj and "metrics" in obj:
                obj["sample_metrics"] = obj.get("metrics")
            samples.append(obj)
            return
        # Nested container with samples
        if "samples" in obj:
            _collect_from_obj(obj["samples"])
        # Some files store a single sample under 'sample_results'
        if "sample_results" in obj and isinstance(obj["sample_results"], dict):
            sr = obj["sample_results"]
            # Build a minimal sample dict when only qa_results/retrieval_results exist
            merged_metrics = {}
            for k in ("qa_results", "retrieval_results", "sample_metrics"):
                part = sr.get(k)
                if isinstance(part, dict):
                    merged_metrics.update(part)
            if merged_metrics:
                s = {
                    "question": sr.get("question") or obj.get("question") or "unknown",
                    "sample_metrics": merged_metrics,
                    "question_metadata": sr.get("metadata") or sr.get("question_metadata") or obj.get("metadata"),
                }
                samples.append(s)
        # Track any overall metrics we see
        if "overall_metrics" in obj and isinstance(obj["overall_metrics"], dict):
            overall_metrics_files.append((None, obj["overall_metrics"]))

    for root, _, files in os.walk(output_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            if not (
                fname.startswith("rag_results_")
                or fname.startswith("overall_results")
                or "rag_results" in fname
                or "samples" in root
            ):
                continue
            fpath = os.path.join(root, fname)
            try:
                data = json.load(open(fpath, "r"))
            except Exception:
                continue
            _collect_from_obj(data)
    return samples, overall_metrics_files


def locomo_category_of(sample: dict) -> str:
    """Map LoCoMo question type code to string category.
    Expected mapping (example provided by author): {2: "single-hop", 3: "temporal", 4: "open-domain", 5: "adversarial", 1: "multi-hop"}
    Falls back to 'unknown' if not present.
    """
    mapping = {2: "single-hop", 3: "temporal", 4: "open-domain", 5: "adversarial", 1: "multi-hop"}
    qm = sample.get("question_metadata") or {}
    qtype = qm.get("type")
    # qtype may be numeric or string
    try:
        code = int(qtype)
        return mapping.get(code, str(qtype))
    except Exception:
        # accept string label directly
        return str(qtype) if qtype is not None else "unknown"


def complextr_category_of(sample: dict) -> str:
    """Classify ComplexTR sample by question_metadata.type prefix: L2 -> time-to-event, L3 -> event-to-event"""
    qm = sample.get("question_metadata") or {}
    # Prefer ID prefix if present
    sid = str(qm.get("id", "") or "")
    if sid.startswith("L2"):
        return "time_to_event"
    if sid.startswith("L3"):
        return "event_to_event"
    t = str(qm.get("type", "") or "")
    if t.startswith("L2"):
        return "time_to_event"
    if t.startswith("L3"):
        return "event_to_event"
    return "unknown"


def aggregate_by_category(samples, dataset: str):
    """Aggregate metrics per category for given dataset."""
    by_cat_sum = defaultdict(lambda: defaultdict(float))
    by_cat_count = defaultdict(int)

    for s in samples:
        metrics = s.get("sample_metrics") or {}
        if not isinstance(metrics, dict):
            continue
        if dataset == "locomo":
            cat = locomo_category_of(s)
        elif dataset == "complex_tr":
            cat = complextr_category_of(s)
        else:
            cat = "unknown"
        by_cat_count[cat] += 1
        for k, v in metrics.items():
            try:
                by_cat_sum[cat][k] += float(v)
            except Exception:
                pass

    # build averages
    by_cat_avg = {}
    for cat, counts in by_cat_count.items():
        if counts <= 0:
            continue
        avgs = {}
        for k, total in by_cat_sum[cat].items():
            avgs[k] = round(total / counts, 4)
        by_cat_avg[cat] = {"n": counts, "metrics": avgs}
    return by_cat_avg


def format_table_locomo(by_cat_avg: dict):
    # Target headers: Avg | Single-Hop | Multi-Hop | Open-Domain | Temporal | Adversarial
    order = [
        ("single-hop", "Single-Hop"),
        ("multi-hop", "Multi-Hop"),
        ("open-domain", "Open-Domain"),
        ("temporal", "Temporal"),
        ("adversarial", "Adversarial"),
    ]
    # Determine averages across all
    overall = by_cat_avg.get("overall")
    # Compose rows for typical metrics if present
    metrics_keys = set()
    for v in by_cat_avg.values():
        metrics_keys.update(v.get("metrics", {}).keys())
    metrics_keys = sorted(metrics_keys)

    lines = []
    header = ["Category", "Overall"] + [name for _, name in order]
    lines.append("\t".join(header))
    for mk in metrics_keys:
        row = [mk]
        overall_cell = by_cat_avg.get("overall", {}).get("metrics", {}).get(mk)
        row.append("-" if overall_cell is None else f"{overall_cell:.4f}")
        for key, _ in order:
            cell = by_cat_avg.get(key, {}).get("metrics", {}).get(mk)
            row.append("-" if cell is None else f"{cell:.4f}")
        lines.append("\t".join(row))
    return "\n".join(lines)


def format_table_complextr(by_cat_avg: dict):
    # Headers: Avg | Time to Event | Event to Event
    order = [("time_to_event", "Time to Event"), ("event_to_event", "Event to Event")]
    metrics_keys = set()
    for v in by_cat_avg.values():
        metrics_keys.update(v.get("metrics", {}).keys())
    metrics_keys = sorted(metrics_keys)
    lines = []
    header = ["Category", "Overall"] + [name for _, name in order]
    lines.append("\t".join(header))
    for mk in metrics_keys:
        row = [mk]
        overall_cell = by_cat_avg.get("overall", {}).get("metrics", {}).get(mk)
        row.append("-" if overall_cell is None else f"{overall_cell:.4f}")
        for key, _ in order:
            cell = by_cat_avg.get(key, {}).get("metrics", {}).get(mk)
            row.append("-" if cell is None else f"{cell:.4f}")
        lines.append("\t".join(row))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Per-category analysis for LoCoMo and ComplexTR")
    parser.add_argument(
        "--output_dir", required=True, help="Top-level outputs directory for the experiment (e.g., outputs/locomo)"
    )
    parser.add_argument("--dataset", required=True, choices=["locomo", "complex_tr"], help="Which dataset to analyze")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional path to save CSV-style table")
    args = parser.parse_args()

    samples, overall_files = load_all_rag_results(args.output_dir)
    if not samples:
        raise SystemExit(
            f"No samples found under {args.output_dir}. For ComplexTR, ensure the folder contains rag_results_*.json or per-sample JSONs under a 'samples/' subdirectory with sample_results/qa_results structures."
        )

    # Add an overall aggregate as well
    by_cat = aggregate_by_category(samples, args.dataset)

    # Also compute overall average across all questions
    overall_sum = defaultdict(float)
    for s in samples:
        sm = s.get("sample_metrics") or {}
        for k, v in sm.items():
            try:
                overall_sum[k] += float(v)
            except Exception:
                pass
    overall_avg = {k: round(v / len(samples), 4) for k, v in overall_sum.items()}
    by_cat["overall"] = {"n": len(samples), "metrics": overall_avg}

    if args.dataset == "locomo":
        table = format_table_locomo(by_cat)
    else:
        table = format_table_complextr(by_cat)

    print("\nPer-category results (approximate, % if metrics already scaled):")
    print(table)

    if args.save_csv:
        try:
            with open(args.save_csv, "w") as f:
                f.write(table.replace("\t", ","))
            print(f"Saved table to {args.save_csv}")
        except Exception as e:
            print(f"Failed to save CSV: {e}")


if __name__ == "__main__":
    main()
