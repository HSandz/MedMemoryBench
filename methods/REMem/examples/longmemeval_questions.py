import argparse
import json
import logging
import os

from examples.longmemeval import preprocess_longmemeval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReMem retrieval and QA")
    args = parser.parse_args()

    print(args)
    samples = json.load(open("reproduce/dataset/longmemeval/longmemeval_s"))
    logging.basicConfig(level=logging.INFO)
    logging.info(f"# of samples: {len(samples)}")

    preprocess_longmemeval(samples)

    # Dictionary to group questions by category
    questions_by_category = {}
    num_sample = 0

    for sample_idx, sample in enumerate(samples):
        num_sample += 1
        question = sample["question"]
        answer = sample["answer"]
        question_type = sample["question_type"]
        question_id = sample["question_id"]
        question_date = sample["question_date"]

        # Group questions by category
        if question_type not in questions_by_category:
            questions_by_category[question_type] = []
        questions_by_category[question_type].append(
            {
                "sample_idx": sample_idx,
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "question_date": question_date,
            }
        )

    # Prepare output data structure
    output_data = {
        "summary": {
            "total_questions": sum(len(questions_list) for questions_list in questions_by_category.values()),
            "total_categories": len(questions_by_category),
            "questions_per_category": {},
        },
        "questions_by_category": {},
    }

    # Process each category
    for question_type, questions_list in questions_by_category.items():
        output_data["summary"]["questions_per_category"][question_type] = len(questions_list)
        output_data["questions_by_category"][question_type] = {
            "category_name": question_type,
            "total_questions": len(questions_list),
            "questions": questions_list,
        }

    # Create output directory if it doesn't exist
    output_dir = "outputs/longmemeval"
    os.makedirs(output_dir, exist_ok=True)

    # Write to JSON file
    output_path = os.path.join(output_dir, "questions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\nQuestion organization complete!")
    print(f"Total questions: {output_data['summary']['total_questions']}")
    print(f"Total categories: {output_data['summary']['total_categories']}")
    print("\nQuestions per category:")
    for category, count in output_data["summary"]["questions_per_category"].items():
        print(f"  {category}: {count}")
    print(f"\nResults saved to {output_path}")

    # Also create the original demos output for backward compatibility
    demos = []
    for sample in samples:
        gold_sessions = [sample["haystack_docs"][answer_idx] for answer_idx in sample["answer_session_idxs"]]
        gold_rounds = []
        for session in gold_sessions:
            session_json = json.loads(session)
            gold_rounds.extend(message for message in session_json["messages"] if message.get("has_answer") is True)
        demos.append({"question": sample["question"], "answer": sample["answer"], "gold_rounds": gold_rounds})

    demos_output_path = os.path.join(output_dir, "longmemeval_s_demos.json")
    with open(demos_output_path, "w") as f:
        json.dump(demos, f, indent=4)
    print(f"Demos also saved to {demos_output_path}")
