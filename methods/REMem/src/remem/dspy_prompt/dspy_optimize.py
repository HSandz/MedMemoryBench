import argparse
import os

import dspy
from dspy import Evaluate

from remem.rerank import Fact


class TripleFilteringSignature(dspy.Signature):
    """
     The user is trying to ask a question about the conversation between the user itself and the assistant.
     Now, some facts (triples) are extracted from their conversation, and you need to filter facts based on their relevance to the question.
     Compare the facts in candidate list, carefully generate related facts from the list that have strong connection to the question.

    - Multi-hop reasoning may be required, meaning you might need to combine multiple facts to form a complete response.
    - The relevance means the triple aids in reasoning and providing an answer.
    - Select relevant facts from the candidate list and output in JSON format without any other words, e.g.,

    ```json
    {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}.
    ```

    - If no fact is relevant, return an empty list, e.g., {"fact": []}.
    - Only use facts from the candidate list; do NOT generate new fact.
    """

    question = dspy.InputField(desc="Query for retrieval")
    fact_before_filter = dspy.InputField(desc="Candidate triples to be filtered")
    fact_after_filter: Fact = dspy.OutputField(desc="Filtered triples in JSON format")


class TripleFilterProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(TripleFilteringSignature)

    def forward(self, question, fact_before_filter):
        try:
            prediction = self.prog(question=question, fact_before_filter=fact_before_filter)
            return prediction
        except Exception as e:
            print(f"Filter forward exception: {e}")
            from dspy.primitives.prediction import Prediction

            return Prediction(fact_after_filter=Fact(fact=[]))


def filtering_precision(example, pred, trace=None):
    try:
        if len(pred) == 0:
            pred_list = []
        else:
            pred_list = pred.fact_after_filter.fact
    except Exception as e:
        print(f"Error: {e}")
        pred_list = []
    try:
        gold_list = example.fact_after_filter
    except Exception as e:
        print(f"Error: {e}")
        gold_list = []

    gold_set = set([tuple(t) for t in gold_list])
    pred_set = set([tuple(t) for t in pred_list])
    if len(pred_set) == 0 and len(gold_set) == 0:
        return 1
    elif len(pred_set) == 0 and len(gold_set) > 0:
        return 0
    elif len(gold_set) == 0 and len(pred_set) > 0:
        return 0

    if trace is None:
        return len(gold_set.intersection(pred_set)) / len(pred_set)
    else:
        return gold_set == pred_set


def filtering_recall(example, pred, trace=None):
    try:
        if len(pred) == 0:
            pred_list = []
        else:
            pred_list = pred.fact_after_filter.fact
    except Exception as e:
        print(f"Error: {e}")
        pred_list = []
    try:
        gold_list = example.fact_after_filter
    except Exception as e:
        print(f"Error: {e}")
        gold_list = []

    gold_set = set([tuple(t) for t in gold_list])
    pred_set = set([tuple(t) for t in pred_list])
    if len(pred_set) == 0 and len(gold_set) == 0:
        return 1
    elif len(pred_set) == 0 and len(gold_set) > 0:
        return 0
    elif len(gold_set) == 0 and len(pred_set) > 0:
        return 1
    return len(gold_set.intersection(pred_set)) / len(gold_set)


def filtering_f1(example, pred, trace=None):
    p = filtering_precision(example, pred, trace)
    r = filtering_recall(example, pred, trace)
    if p + r == 0:
        return 0
    f1 = 2 * p * r / (p + r)
    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="Default DSPy LM")
    parser.add_argument("--addr", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="8000")
    parser.add_argument(
        "--auto", type=str, default="heavy", help="Optimization level", choices=["light", "medium", "heavy"]
    )
    parser.add_argument("--max_demos", type=int, default=10)
    parser.add_argument("--teacher", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument(
        "--train",
        type=str,
        help="Path to training set",
        default="src/remem/prompts/dspy_prompts/musique_2wiki_train_1000.json",
    )
    parser.add_argument(
        "--dev",
        type=str,
        help="Path to dev set",
        default="src/remem/prompts/dspy_prompts/longmemeval_triple_filter_dev_10.json",
    )
    args = parser.parse_args()

    THREAD = 24
    print(args)

    if args.llm.startswith("gpt-") or args.llm.startswith("ft:gpt-") or args.llm.startswith("o1-"):
        dspy_llm = dspy.LM(model=f"openai/{args.llm}", max_tokens=2000, temperature=0.0)
    elif args.addr is not None and args.port is not None and "llama" in args.llm.lower():
        url = f"http://{args.addr}:{args.port}/v1"
        dspy_llm = dspy.LM(
            model=f"hosted_vllm/{args.llm}", max_tokens=2000, temperature=0.0, api_base=url, api_key="osunlp"
        )
    else:
        raise ValueError(f"LM not implemented: {args.llm}")
    dspy.settings.configure(lm=dspy_llm)

    if args.teacher is None:
        args.teacher = args.llm

    if args.teacher.startswith("gpt-") or args.teacher.startswith("ft:gpt-"):
        teacher_lm = dspy.LM(model=f"openai/{args.teacher}", max_tokens=2000, temperature=0.0)
    elif args.teacher.startswith("o1-"):
        teacher_lm = dspy.LM(model=f"openai/{args.teacher}", max_tokens=5000, temperature=1.0)
    elif args.addr is not None and args.port is not None and "llama" in args.teacher.lower():
        url = f"http://{args.addr}:{args.port}/v1"
        teacher_lm = dspy.LM(
            model=f"hosted_vllm/{args.teacher}", max_tokens=2000, temperature=0.0, api_base=url, api_key="osunlp"
        )
    else:
        raise NotImplementedError(f"Teacher model {args.teacher} not implemented yet.")

    from dspy.datasets import DataLoader

    dataloader = DataLoader()
    train_set = dataloader.from_json(
        args.train,
        fields=("question", "fact_before_filter", "fact_after_filter"),
        input_keys=("question", "fact_before_filter"),
    )
    dev_set = dataloader.from_json(
        args.dev,
        fields=("question", "fact_before_filter", "fact_after_filter"),
        input_keys=("question", "fact_before_filter"),
    )

    filter_metric = filtering_f1
    evaluate = Evaluate(
        devset=dev_set[:], metric=filter_metric, num_threads=THREAD, display_progress=True, display_table=False
    )

    # Initialize optimizer
    prompt_lm = dspy_llm
    kwargs = dict(num_threads=THREAD, teacher_settings=dict(lm=teacher_lm), prompt_model=prompt_lm)
    optimizer = dspy.MIPROv2(metric=filter_metric, auto=args.auto, **kwargs)
    program = TripleFilterProgram()

    # Optimize program
    print("Optimizing program with MIPRO...")
    optimized_program = optimizer.compile(
        program.deepcopy(),
        trainset=train_set,
        valset=dev_set,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos,
        requires_permission_to_run=False,
    )

    # Save optimize program for future use
    os.makedirs("outputs/dspy", exist_ok=True)
    model_label = args.llm.replace("/", "_")
    teacher_model_label = args.teacher.replace("/", "_")
    output_path = f"outputs/dspy/triple_filter_mipro_optimized_{model_label}_teacher_{teacher_model_label}.json"
    optimized_program.save(output_path)

    # Evaluate optimized program
    print("Evaluate optimized program...")
    evaluate(optimized_program, devset=dev_set[:])
    print(f"Optimized program saved at {output_path}")
