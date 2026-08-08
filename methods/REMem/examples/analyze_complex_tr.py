import argparse
import json
import os
from collections import defaultdict


def is_query_solution_dict(d: dict) -> bool:
    return isinstance(d, dict) and (
        ("question" in d and ("sample_metrics" in d or "metrics" in d))
        or ("sample_results" in d and isinstance(d["sample_results"], dict))
        or ("qa_results" in d and isinstance(d["qa_results"], dict))
    )


def harvest_samples_from_json(obj, out_list):
    if isinstance(obj, list):
        for item in obj:
            harvest_samples_from_json(item, out_list)
        return
    if not isinstance(obj, dict):
        return

    # Direct QuerySolution style
    if "question" in obj and ("sample_metrics" in obj or "metrics" in obj):
        if "sample_metrics" not in obj and "metrics" in obj:
            obj["sample_metrics"] = obj.get("metrics")
        out_list.append(obj)
        return

    # Newer overall format: {'sample': {...}} as in tot or others
    if "sample" in obj and isinstance(obj["sample"], dict):
        s = obj["sample"]
        merged = dict(s)
        if "qa_results" in obj and isinstance(obj["qa_results"], dict):
            merged.setdefault("sample_metrics", {}).update(obj["qa_results"])
        if "retrieval_results" in obj and isinstance(obj["retrieval_results"], dict):
            merged.setdefault("sample_metrics", {}).update(obj["retrieval_results"])
        # Ensure presence of sample_metrics
        if "sample_metrics" not in merged:
            # Try inside s
            sm = merged.get("qa_results", {})
            sm.update(merged.get("retrieval_results", {}))
            if sm:
                merged["sample_metrics"] = sm
        out_list.append(merged)

    # ComplexTR per-sample wrapper: {'sample_results': {...}}
    if "sample_results" in obj and isinstance(obj["sample_results"], dict):
        sr = obj["sample_results"]
        s = {
            "question": sr.get("question", "unknown"),
            "question_metadata": sr.get("metadata") or sr.get("question_metadata"),
            "sample_metrics": {},
        }
        for k in ("qa_results", "retrieval_results", "sample_metrics"):
            if isinstance(sr.get(k), dict):
                s["sample_metrics"].update(sr[k])
        if s["sample_metrics"]:
            out_list.append(s)

    # Nested containers
    for key in ("samples", "data", "items"):
        if key in obj:
            harvest_samples_from_json(obj[key], out_list)


def load_complex_tr_samples(output_dir: str):
    samples = []
    json_files = 0
    for root, _, files in os.walk(output_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            # focus on likely result files
            if not (
                fname.startswith("rag_results")
                or fname.startswith("overall_results")
                or "sample" in fname
                or "rag_results" in fname
            ):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                json_files += 1
            except Exception:
                continue
            # Some overall files store samples as list of lists with single dict inside
            if (
                isinstance(data, dict)
                and isinstance(data.get("samples"), list)
                and data["samples"]
                and isinstance(data["samples"][0], list)
            ):
                flat = []
                for row in data["samples"]:
                    if isinstance(row, list):
                        flat.extend([d for d in row if isinstance(d, dict)])
                harvest_samples_from_json({"samples": flat}, samples)
            else:
                harvest_samples_from_json(data, samples)
    return samples, json_files


def complextr_category_of(sample: dict, idx_to_type: dict = None, id_to_type: dict = None) -> str:
    qm = sample.get("question_metadata") or {}
    t = str(qm.get("type", "") or "")
    # Strongest signal: ID prefix like 'L2_...' or 'L3_...'
    sid = str(qm.get("id", "") or sample.get("id", "") or "")
    if sid.startswith("L2"):
        return "time_to_event"
    if sid.startswith("L3"):
        return "event_to_event"
    if not t and idx_to_type:
        try:
            sidx = qm.get("sample_idx")
            if sidx is None and isinstance(sample.get("sample_idx"), int):
                sidx = sample.get("sample_idx")
            if sidx is not None and sidx in idx_to_type:
                t = str(idx_to_type[sidx] or "")
        except Exception:
            pass
    if not t and id_to_type:
        try:
            sid = qm.get("id") or sample.get("id")
            if sid is not None and sid in id_to_type:
                t = str(id_to_type[sid] or "")
        except Exception:
            pass
    if not t:
        # heuristic: sometimes question string begins with L2/L3
        q = str(sample.get("question", "") or "")
        if q.strip().startswith("L2"):
            t = "L2"
        elif q.strip().startswith("L3"):
            t = "L3"
    if t.startswith("L2"):
        return "time_to_event"
    if t.startswith("L3"):
        return "event_to_event"
    return "unknown"


def aggregate(samples, idx_to_type: dict = None, id_to_type: dict = None):
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for s in samples:
        cat = complextr_category_of(s, idx_to_type, id_to_type)
        metrics = s.get("sample_metrics") or {}
        if not isinstance(metrics, dict):
            continue
        counts[cat] += 1
        for k, v in metrics.items():
            try:
                sums[cat][k] += float(v)
            except Exception:
                pass
    avgs = {}
    for cat, n in counts.items():
        if n <= 0:
            continue
        avgs[cat] = {"n": n, "metrics": {k: round(total / n, 4) for k, total in sums[cat].items()}}
    # overall
    overall_totals = defaultdict(float)
    overall_n = 0
    for s in samples:
        m = s.get("sample_metrics") or {}
        if not isinstance(m, dict):
            continue
        overall_n += 1
        for k, v in m.items():
            try:
                overall_totals[k] += float(v)
            except Exception:
                pass
    if overall_n > 0:
        avgs["overall"] = {"n": overall_n, "metrics": {k: round(v / overall_n, 4) for k, v in overall_totals.items()}}
    return avgs


def format_table(avgs):
    order = [("time_to_event", "Time to Event"), ("event_to_event", "Event to Event")]
    metrics_keys = set()
    for v in avgs.values():
        metrics_keys.update((v.get("metrics") or {}).keys())
    metrics_keys = sorted(metrics_keys)
    lines = []
    header = ["Category"] + [name for _, name in order]
    lines.append("\t".join(header))
    for mk in metrics_keys:
        row = [mk]
        for key, _ in order:
            cell = avgs.get(key, {}).get("metrics", {}).get(mk)
            row.append("-" if cell is None else f"{cell:.2f}")
        lines.append("\t".join(row))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze ComplexTR per-category results")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--dataset_json",
        type=str,
        default="reproduce/dataset/complex-tr/complex_tr_1000.json",
        help="Path to ComplexTR dataset JSON to recover type (L2/L3)",
    )
    parser.add_argument("--save_csv", type=str, default=None)
    args = parser.parse_args()

    # Build index -> type map from dataset JSON (many runs omit type in saved metadata)
    idx_to_type = {}
    id_to_type = {}
    q_to_type = {}
    if args.dataset_json and os.path.exists(args.dataset_json):
        try:
            raw = json.load(open(args.dataset_json, "r"))
            if isinstance(raw, list):
                for i, s in enumerate(raw):
                    if not isinstance(s, dict):
                        continue
                    t = s.get("type")
                    if t is not None:
                        idx_to_type[i] = t
                    sid = s.get("id")
                    if sid is not None:
                        id_to_type[sid] = t
                    qtext = s.get("question") or s.get("query")
                    if isinstance(qtext, str) and t is not None:
                        q_to_type[qtext.strip().lower()] = t
            elif isinstance(raw, dict) and "samples" in raw and isinstance(raw["samples"], list):
                for i, s in enumerate(raw["samples"]):
                    if not isinstance(s, dict):
                        continue
                    if "type" in s:
                        idx_to_type[i] = s["type"]
                    sid = s.get("id")
                    if sid is not None:
                        id_to_type[sid] = s.get("type")
                    qtext = s.get("question") or s.get("query")
                    if isinstance(qtext, str) and ("type" in s):
                        q_to_type[qtext.strip().lower()] = s["type"]
        except Exception:
            idx_to_type = {}
            id_to_type = {}

    samples, scanned = load_complex_tr_samples(args.output_dir)
    if not samples:
        raise SystemExit(
            f"No ComplexTR samples found under {args.output_dir}. Scanned {scanned} JSON files. Ensure per-sample JSONs or an overall_results file exist."
        )
    # Fill missing types using question text matching
    if q_to_type:
        for s in samples:
            qm = s.get("question_metadata") or {}
            t = qm.get("type")
            if not t:
                q = (s.get("question") or "").strip().lower()
                if q in q_to_type:
                    if isinstance(s.setdefault("question_metadata", {}), dict):
                        s["question_metadata"]["type"] = q_to_type[q]
    avgs = aggregate(samples, idx_to_type, id_to_type)
    table = format_table(avgs)
    print("\nPer-category results (ComplexTR):")
    print(table)
    if args.save_csv:
        with open(args.save_csv, "w") as f:
            f.write(table.replace("\t", ","))
        print(f"Saved CSV to {args.save_csv}")


if __name__ == "__main__":
    main()
