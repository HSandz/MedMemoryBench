import json
from collections import defaultdict

if __name__ == "__main__":
    samples = json.load(open("reproduce/dataset/longmemeval/longmemeval_s"))

    question_type_to_qa_pairs = defaultdict(list)
    for sample_idx, sample in enumerate(samples):
        question_type = sample["question_type"]
        question = sample["question"]
        answer = sample["answer"]
        question_type_to_qa_pairs[question_type].append((question, answer))

    print(json.dumps(question_type_to_qa_pairs, indent=2))

    # print the number of samples for each type
    for question_type, qa_pairs in question_type_to_qa_pairs.items():
        print(question_type, len(qa_pairs))
