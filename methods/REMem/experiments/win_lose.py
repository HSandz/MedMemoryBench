import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log1', type=str)
    parser.add_argument('--log2', type=str)
    parser.add_argument('--retrieval_metric', type=str, default="Recall_all@5")
    parser.add_argument('--qa_metric', type=str, default="LLMEval")
    parser.add_argument('--label', type=str)
    args = parser.parse_args()

    with open(args.log1, 'r') as f:
        log1 = json.load(f)
    with open(args.log2, 'r') as f:
        log2 = json.load(f)

    # Sort two logs by sample_idx
    assert len(log1["samples"]) == len(log2["samples"]), "Two logs have different number of samples"
    log1["samples"].sort(key=lambda x: x["sample_idx"])
    log2["samples"].sort(key=lambda x: x["sample_idx"])

    compare = {"retrieval_win": [], "retrieval_lose": [], "retrieval_tie": [], "qa_win": [], "qa_lose": [], "qa_tie": []}
    # Compare the two logs
    for idx in range(len(log1["samples"])):
        sample1 = log1["samples"][idx]
        sample2 = log2["samples"][idx]

        assert sample1["sample_idx"] == sample2["sample_idx"], "Sample index mismatch"

        retrieval1 = sample1["sample_metrics"][args.retrieval_metric]
        retrieval2 = sample2["sample_metrics"][args.retrieval_metric]
        qa1 = sample1["sample_metrics"][args.qa_metric]
        qa2 = sample2["sample_metrics"][args.qa_metric]

        compared_sample = sample1
        compared_sample["predicted_answer1"] = sample1["predicted_answer"]
        compared_sample["predicted_answer2"] = sample2["predicted_answer"]
        compared_sample["sample_metrics1"] = sample1["sample_metrics"]
        compared_sample["sample_metrics2"] = sample2["sample_metrics"]
        compared_sample["graph_seeds1"] = sample1["graph_seeds"]
        compared_sample["graph_seeds2"] = sample2["graph_seeds"]
        compared_sample["retrieved_chunks1"] = sample1["retrieved_chunks"]
        compared_sample["retrieved_chunks2"] = sample2["retrieved_chunks"]
        del compared_sample["retrieved_chunks"]
        del compared_sample["graph_seeds"]
        del compared_sample["sample_metrics"]
        del compared_sample["predicted_answer"]

        if retrieval1 > retrieval2:
            compare["retrieval_win"].append(compared_sample)
        elif retrieval1 < retrieval2:
            compare["retrieval_lose"].append(compared_sample)
        else:
            compare["retrieval_tie"].append(compared_sample)
        if qa1 > qa2:
            compare["qa_win"].append(compared_sample)
        elif qa1 < qa2:
            compare["qa_lose"].append(compared_sample)
        else:
            compare["qa_tie"].append(compared_sample)

    # write comparison results to file
    output_path = f"experiments/{str(args.label)  + '_' if args.label else ''}comparison.json"
    with open(output_path, 'w') as f:
        json.dump(compare, f, indent=4)
        print("Comparison results saved to", output_path)

    # print the length of each list in `compare`
    print("Retrieval win:", len(compare["retrieval_win"]))
    print("Retrieval lose:", len(compare["retrieval_lose"]))
    print("Retrieval tie:", len(compare["retrieval_tie"]))
    print("QA win:", len(compare["qa_win"]))
    print("QA lose:", len(compare["qa_lose"]))
    print("QA tie:", len(compare["qa_tie"]))