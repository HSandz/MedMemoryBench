import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="outputs", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, help="LLM name")
    args = parser.parse_args()

    llm_label = args.llm_name.replace("/", "_")
    all_openie_results = []
    locomo_dir = os.path.join(args.log, "locomo")
    for dir_name in os.listdir(locomo_dir):
        if not dir_name.startswith("locomo_"):
            continue
        if "dpr" in dir_name:
            continue

        openie_results_path = f"{locomo_dir}/{dir_name}/openie_results_ner_{llm_label}.json"
        if os.path.isfile(openie_results_path):
            with open(openie_results_path, "r") as f:
                openie_results = json.load(f)
                all_openie_results.extend(openie_results["docs"])

    output_path = f"outputs/locomo/openie_results_{llm_label}.json"
    with open(output_path, "w") as f:
        json.dump(all_openie_results, f, indent=4)
        print("Saved OpenIE results to", output_path)
