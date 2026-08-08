import argparse
import json
import os

from tqdm import tqdm

from examples.locomo import get_candidate_messages, get_gold_docs_for_qa_pair

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    samples = json.load(open("reproduce/dataset/locomo/locomo10.json"))

    # Dictionary to group questions by category
    questions_by_category = {}
    num_sample = 0

    for sample_idx, sample in tqdm(enumerate(samples), total=len(samples)):
        conversation = sample["conversation"]
        qa_pairs = sample["qa"]
        observation = sample["observation"]
        candidate_docs = get_candidate_messages(conversation)
        print(f"sample {sample_idx} # chunk contents:", len(candidate_docs))

        dataset_name = f"locomo_{sample_idx}"

        query_solutions = []
        questions = []
        gold_docs = []
        gold_answers = []
        question_metadata = []

        for qa_idx, qa_pair in enumerate(qa_pairs):
            num_sample += 1
            question = qa_pair["question"]
            category = qa_pair["category"]
            questions.append(question)

            if "answer" not in qa_pair:
                cur_gold_answers = ["no information available"]
            else:
                cur_gold_answers = [str(qa_pair["answer"])]
            gold_answers.append(cur_gold_answers)
            cur_gold_docs = []
            question_metadata.append({"type": category})

            new_evidence_list = []
            for evidence in qa_pair["evidence"]:
                if "; " in evidence:
                    new_evidence_list.extend(evidence.split("; "))
                else:
                    new_evidence_list.append(evidence)
            qa_pair["evidence"] = new_evidence_list

            cur_gold_docs = get_gold_docs_for_qa_pair(qa_pair, conversation)
            gold_docs.append(cur_gold_docs)

            # Group questions by category
            if category not in questions_by_category:
                questions_by_category[category] = []
            questions_by_category[category].append(
                {
                    "sample_idx": sample_idx,
                    "qa_idx": qa_idx,
                    "question": question,
                    "answer": qa_pair["answer"] if "answer" in qa_pair else None,
                }
            )

        # end for each QA pair

    # Prepare output data structure
    category_num_to_str = {
        1: "multi-hop",
        2: "single-hop",
        3: "temporal-reasoning",
        4: "open-domain knowledge",
        5: "abstention",
    }

    output_data = {
        "summary": {
            "total_questions": sum(len(questions_list) for questions_list in questions_by_category.values()),
            "total_categories": len(questions_by_category),
            "questions_per_category": {},
        },
        "questions_by_category": {},
    }

    # Process each category
    for category, questions_list in questions_by_category.items():
        category_name = category_num_to_str[category]
        output_data["summary"]["questions_per_category"][category_name] = len(questions_list)
        output_data["questions_by_category"][category_name] = {
            "category_id": category,
            "category_name": category_name,
            "total_questions": len(questions_list),
            "questions": questions_list,
        }

    # Create output directory if it doesn't exist
    output_dir = "outputs/locomo"
    os.makedirs(output_dir, exist_ok=True)

    # Write to JSON file
    output_path = os.path.join(output_dir, "questions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Questions grouped by category saved to: {output_path}")
    print(f"Total questions: {output_data['summary']['total_questions']}")
    print(f"Total categories: {output_data['summary']['total_categories']}")
    for category_name, count in output_data["summary"]["questions_per_category"].items():
        print(f"  {category_name}: {count} questions")
