import argparse
import json
import os
from collections import defaultdict

from tqdm import tqdm

from remem.remem import ReMem, logger
from remem.utils.config_utils import BaseConfig
from remem.utils.misc_utils import safe_dump_json, safe_serialize_query_solutions
from remem.utils.results_utils import get_inference_type, get_working_dir


def get_candidate_messages(conversation):
    res = []
    for key in conversation:
        if key.startswith("session_") and not key.endswith("date_time"):  # session_{idx}
            for dialog in conversation[key]:
                speaker = dialog["speaker"]
                text = dialog["text"]
                dialog_id = dialog["dia_id"]
                date = conversation[f"{key}_date_time"]
                res.append(json.dumps({"messages": [{"content": text, "role": speaker, "date": date}]}))
    return res


def get_sessions(conversation):
    sessions = []
    for key in conversation:
        if key.startswith("session_") and not key.endswith("date_time"):  # session_{idx}
            session = []
            for dialog in conversation[key]:
                speaker = dialog["speaker"]
                text = dialog["text"]
                dialog_id = dialog["dia_id"]
                date = conversation[f"{key}_date_time"]
                try:
                    session_idx, message_idx = map(int, dialog_id.lstrip("D").split(":"))
                except Exception:
                    session_idx = None
                    message_idx = None
                session.append(
                    {
                        "role": speaker,
                        "content": text,
                        "date": date,
                        "dialog_id": dialog_id,
                        "session_idx": session_idx,
                        "message_idx": message_idx,
                    }
                )
            sessions.append(session)
    return sessions


def get_gold_docs_for_qa_pair(qa_pair, conversation):
    gold_docs = []
    for evidence in qa_pair["evidence"]:
        try:
            evidence_session_idx = int(evidence[1:].split(":")[0])
            evidence_dialogue_idx = int(evidence[1:].split(":")[1])
            dialog = conversation[f"session_{evidence_session_idx}"][evidence_dialogue_idx - 1]
            content = f"{dialog['speaker']}: {dialog['text']}"
            session_time = conversation.get(f"session_{evidence_session_idx}_date_time", None)
            gold_docs.append(json.dumps({"messages": [{"content": content, "date": session_time}]}))
        except Exception as e:
            logger.error("Gold doc not found: %s", e)
            logger.error("Evidence: %s", evidence)
    return gold_docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4.1-mini-2025-04-14", help="LLM name")
    parser.add_argument("--dataset", type=str, default="locomo_episodic")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2", help="embedding model name")
    parser.add_argument("-fi", "--force_index_from_scratch", action="store_true")
    parser.add_argument("-fo", "--force_openie_from_scratch", action="store_true")
    parser.add_argument("-fr", "--force_rag", action="store_true", help="Force rerun RAG even if results exist")
    parser.add_argument(
        "--llm_infer_mode",
        choices=["online", "offline"],
        default="online",
        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes",
    )
    parser.add_argument("--use_azure", action="store_true", help="Use Azure for OPENAI")
    parser.add_argument("--indices", type=str)
    parser.add_argument("--extract_format", type=str)
    parser.add_argument("--extract_method", type=str, default="episodic_gist")
    parser.add_argument("--qa_top_k", type=int, default=10)
    # Agent configuration parameters
    parser.add_argument(
        "--agent_fixed_tools", action="store_true", help="Use simple agent with only semantic_retrieve + output_answer"
    )
    parser.add_argument(
        "--agent_max_steps",
        type=int,
        default=3,
        help="Maximum reasoning steps for agent (1=retrieve only, 2=retrieve+answer, >2=full reasoning)",
    )
    args = parser.parse_args()

    force_index_from_scratch = args.force_index_from_scratch
    force_openie_from_scratch = args.force_openie_from_scratch
    force_rag_from_scratch = args.force_rag

    dataset_label = "locomo" if args.dataset in ["locomo", "locomo10"] else args.dataset
    samples = json.load(open(f"reproduce/dataset/locomo/{args.dataset}.json"))
    logger.info("# sessions: %d", len(samples))
    logger.info("# questions: %d", sum(len(sample["qa"]) for sample in samples))

    if args.indices is not None:
        indices = json.load(open(args.indices))
        samples = [samples[i] for i in indices]

    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    llm_label = args.llm_name.replace("/", "_") if llm_name is not None else "None"
    embedding_label = args.embedding_name.replace("/", "_") if args.embedding_name is not None else "None"

    config = BaseConfig(
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=dataset_label,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/remem/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=args.qa_top_k,
        do_eval_retrieval=True,
        do_eval_qa=True,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=16,
        max_new_tokens=None,
        corpus_len=None,
        llm_infer_mode=args.llm_infer_mode,
        preprocess_chunk_func="by_session",
        use_azure=args.use_azure,
        extract_format=args.extract_format,
        extract_method=args.extract_method,
        qa_passage_prefix="",
        qa_prompt_template="rag_qa_unified",
        # Agent configuration (only applies to episodic/temporal methods)
        agent_fixed_tools=args.agent_fixed_tools,
        agent_max_steps=args.agent_max_steps,
    )
    if args.llm_infer_mode == "offline":  # the offline vllm instance is used for all RAG instances later
        from remem.llm.vllm_offline import VLLMOffline

        llm_client = VLLMOffline(
            config,
            model_name=args.llm_name,
            cache_dir="outputs/locomo/llm_cache",
        )
    else:
        llm_client = None

    from remem.embedding_model import _get_embedding_client

    embedding_model = _get_embedding_client(
        global_config=config, embedding_model_name=args.embedding_name, openai_style_server=True
    )

    cumulative_metrics = defaultdict(float)
    selected_metrics = ("qa_em", "retrieval_recall_locomo", "qa_f1_score_locomo", "qa_mem0_llm_judge", "qa_bleu1")

    total_metrics = defaultdict(float)
    num_sample = 0
    for sample_idx, sample in tqdm(enumerate(samples), total=len(samples)):
        conversation = sample["conversation"]
        qa_pairs = sample["qa"]
        observation = sample["observation"]
        # candidate_docs = get_candidate_messages(conversation)
        candidate_sessions = get_sessions(conversation)

        logger.info("sample %d # chunk contents: %d", sample_idx, len(candidate_sessions))

        dataset_name = f"{dataset_label}_{sample_idx}"
        config.dataset = dataset_name
        config.__post_init__()

        working_dir = get_working_dir(config.save_dir, dataset_name, llm_label, embedding_label, args.extract_method)
        inference_type = get_inference_type(args.agent_fixed_tools, args.agent_max_steps, args.extract_method)
        rag_results_path = os.path.join(working_dir, f"rag_results_{inference_type}.json")

        # Check if rag_results.json already exists and skip if not forcing rerun
        if os.path.exists(rag_results_path) and not force_rag_from_scratch:
            logger.info("RAG results already exist for sample %d, path %s, skipping...", sample_idx, rag_results_path)
            try:
                with open(rag_results_path, "r") as f:
                    existing_results = json.load(f)
                    if "overall_metrics" in existing_results:
                        # Add existing metrics to total
                        existing_metrics = existing_results["overall_metrics"]
                        num_questions = len(qa_pairs)
                        for key in existing_metrics:
                            total_metrics[key] += existing_metrics[key] * num_questions
                        num_sample += num_questions
                        logger.info("Loaded existing metrics for sample %d", sample_idx)

                        logger.info("Current cumulative metrics:")
                        for key in total_metrics:
                            logger.info("%s: %.4f", key, round(total_metrics[key] / num_sample, 4))
            except Exception as e:
                logger.error("Error loading existing results for sample %d: %s", sample_idx, e)
                logger.info("Will rerun this sample...")
            else:
                continue

        rag = ReMem(global_config=config, working_dir=working_dir, llm=llm_client)
        rag.set_embedding_model(embedding_model)

        rag.index(candidate_sessions)

        session_metrics = defaultdict(float)
        query_solutions = []
        questions = []
        gold_docs = []
        gold_answers = []
        question_metadata = []
        for qa_idx, qa_pair in enumerate(qa_pairs):
            question = qa_pair["question"]
            questions.append(question)
            if "answer" not in qa_pair:
                cur_gold_answers = ["no information available"]
            else:
                cur_gold_answers = [str(qa_pair["answer"])]
            gold_answers.append(cur_gold_answers)
            cur_gold_docs = []
            question_metadata.append({"type": qa_pair["category"], "temporal_category": qa_pair["temporal_category"]})

            new_evidence_list = []
            for evidence in qa_pair["evidence"]:
                if "; " in evidence:
                    new_evidence_list.extend(evidence.split("; "))
                else:
                    new_evidence_list.append(evidence)
            qa_pair["evidence"] = new_evidence_list

            cur_gold_docs = get_gold_docs_for_qa_pair(qa_pair, conversation)
            gold_docs.append(cur_gold_docs)
        # end for each QA pair

        qa_evaluators, retrieval_evaluators = rag.get_evaluators(gold_answers, gold_docs, selected_metrics)
        (query_solutions, all_response_message, all_metadata, session_retrieval_metrics, session_qa_metrics) = (
            rag.rag_for_qa(questions, gold_docs, gold_answers, selected_metrics, question_metadata=question_metadata)
        )

        num_sample += len(questions)  # Add the count of questions processed for this sample

        for key in session_retrieval_metrics:
            total_metrics[key] += session_retrieval_metrics[key] * len(questions)
            session_metrics[key] += session_retrieval_metrics[key] * len(questions)
        for key in session_qa_metrics:
            total_metrics[key] += session_qa_metrics[key] * len(questions)
            session_metrics[key] += session_qa_metrics[key] * len(questions)

        # print current metrics
        for key in session_retrieval_metrics:
            logger.info("%s: %.4f", key, round(total_metrics[key] / num_sample, 4))
        for key in session_qa_metrics:
            logger.info("%s: %.4f", key, round(total_metrics[key] / num_sample, 4))

        # save RAG results to file
        session_metrics = {key: round(session_metrics[key] / len(questions), 4) for key in session_metrics}
        samples_dict = safe_serialize_query_solutions(query_solutions)
        rag_results = {"samples": samples_dict, "overall_metrics": session_metrics}

        rag_results_path = f"{rag.working_dir}/rag_results_{inference_type}.json"
        success = safe_dump_json(rag_results, rag_results_path)
        if not success:
            logger.warning(f"Warning: Had to use fallback serialization for {rag_results_path}")
    # end for each graph sample (each graph has multiple QA pairs)

    logger.info(f"Total QA pairs: {num_sample}")
