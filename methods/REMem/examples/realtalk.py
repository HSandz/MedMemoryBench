import argparse
import json
import os
import sys
from collections import defaultdict
from glob import glob

from tqdm import tqdm

from remem.remem import ReMem, logger
from remem.utils.config_utils import BaseConfig
from remem.utils.misc_utils import safe_dump_json, safe_serialize_query_solutions
from remem.utils.results_utils import get_inference_type, get_working_dir


def get_candidate_messages_realtalk(conversation):
    """
    Build per-message candidate docs for REALTALK conversations.
    Output format matches ReMem expectations: list of JSON strings, each like
      {"messages": [{"content": <text>, "role": <speaker>, "date": <date_time>}]}.
    """
    res = []
    for key, value in conversation.items():
        if key.startswith("session_") and not key.endswith("date_time") and isinstance(value, list):
            session_date = conversation.get(f"{key}_date_time")
            for dialog in value:
                speaker = dialog.get("speaker")
                text = dialog.get("clean_text") or dialog.get("text") or ""
                date = session_date or dialog.get("date_time")
                res.append(json.dumps({"messages": [{"content": text, "role": speaker, "date": date}]}))
    return res


def map_question_category(cat_value):
    """Map dataset numeric category to label.
    1 -> "multi-hop", 2 -> "temporal_reasoning", 3 -> "commonsense".
    Returns None if missing/unknown.
    """
    try:
        code = int(cat_value)
    except Exception:
        if isinstance(cat_value, str) and cat_value.strip():
            return cat_value.strip()
        return None
    mapping = {1: "multi-hop", 2: "temporal_reasoning", 3: "commonsense"}
    return mapping.get(code)


def get_sessions(conversation):
    """
    Build candidate sessions from REALTALK conversation JSON.
    REALTALK format:
      - Keys like "session_1", "session_2", ... each a list of messages
      - Each message has fields: clean_text, speaker, date_time, dia_id (e.g., "D1:1")
      - Conversation also includes keys like "session_1_date_time"
    We return a list of sessions, each a list of dicts with role/content/date/dialog_id/session_idx/message_idx
    """
    sessions = []
    for key, value in conversation.items():
        if key.startswith("session_") and not key.endswith("date_time") and isinstance(value, list):
            session_idx = None
            try:
                session_idx = int(key.split("_")[1])
            except Exception:
                pass
            session_date = conversation.get(f"{key}_date_time")
            session = []
            for dialog in value:
                speaker = dialog.get("speaker")
                text = dialog.get("clean_text") or dialog.get("text") or ""
                dialog_id = dialog.get("dia_id")
                date_time = session_date or dialog.get("date_time")
                msg_sess_idx = None
                msg_idx = None
                if isinstance(dialog_id, str) and dialog_id.startswith("D") and ":" in dialog_id:
                    try:
                        msg_sess_idx, msg_idx = map(int, dialog_id.lstrip("D").split(":"))
                    except Exception:
                        msg_sess_idx, msg_idx = None, None
                session.append(
                    {
                        "role": speaker,
                        "content": text,
                        "date": date_time,
                        "dialog_id": dialog_id,
                        "session_idx": msg_sess_idx if msg_sess_idx is not None else session_idx,
                        "message_idx": msg_idx,
                    }
                )
            sessions.append(session)
    # sort sessions by session_idx if present
    sessions.sort(key=lambda s: (s[0].get("session_idx") if s and s[0].get("session_idx") is not None else 10**9))
    return sessions


def get_gold_docs_for_qa_pair(qa_pair, conversation):
    """
    Map evidence like "D1:5" to a gold document string with content and date.
    REALTALK messages are in conversation[f"session_{i}"][j-1] with clean_text and speaker.
    """
    gold_docs = []
    for evidence in qa_pair.get("evidence", []):
        try:
            # evidence can contain multiple refs in one string separated by '; '
            refs = [r.strip() for r in str(evidence).split(";")] if ";" in str(evidence) else [str(evidence)]
            for ref in refs:
                if not ref:
                    continue
                evidence_session_idx = int(ref[1:].split(":")[0])
                dialog_list = conversation.get(f"session_{evidence_session_idx}")
                for dialog in dialog_list:
                    if dialog.get("dia_id").strip() == ref.strip():
                        content = f"{dialog.get('speaker')}: {dialog.get('clean_text') or dialog.get('text') or ''}"
                        session_time = conversation.get(
                            f"session_{evidence_session_idx}_date_time", dialog.get("date_time")
                        )
                        gold_docs.append(json.dumps({"messages": [{"content": content, "date": session_time}]}))
                        break
        except Exception as e:
            logger.error("Gold doc not found: %s", e)
            logger.error("Evidence: %s", evidence)
    return gold_docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4.1-mini-2025-04-14", help="LLM name")
    parser.add_argument(
        "--dataset_dir", type=str, default="reproduce/dataset/realtalk", help="Path to REALTALK data directory"
    )
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
    parser.add_argument("--extract_format", type=str)
    parser.add_argument("--extract_method", type=str, default="episodic_gist")
    parser.add_argument("--qa_top_k", type=int, default=10)
    # Agent configuration parameters
    parser.add_argument(
        "--agent_fixed_tools", action="store_true", help="Use simple agent with only semantic_retrieve + output_answer"
    )
    parser.add_argument("--agent_max_steps", type=int, default=3, help="Maximum reasoning steps for agent")
    args = parser.parse_args()

    # Load REALTALK files
    files = sorted(glob(os.path.join(args.dataset_dir, "Chat_*_*.json")))
    if not files:
        logger.error("No REALTALK Chat_*.json files found under %s", args.dataset_dir)
        sys.exit(1)

    force_index_from_scratch = args.force_index_from_scratch
    force_openie_from_scratch = args.force_openie_from_scratch
    force_rag_from_scratch = args.force_rag

    dataset_label = "realtalk"

    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    llm_label = args.llm_name.replace("/", "_") if llm_name is not None else "None"
    embedding_label = args.embedding_name.replace("/", "_") if args.embedding_name is not None else "None"

    config = BaseConfig(
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=dataset_label,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,
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
        agent_fixed_tools=args.agent_fixed_tools,
        agent_max_steps=args.agent_max_steps,
    )

    if args.llm_infer_mode == "offline":
        from remem.llm.vllm_offline import VLLMOffline

        llm_client = VLLMOffline(
            config,
            model_name=args.llm_name,
            cache_dir="outputs/realtalk/llm_cache",
        )
    else:
        llm_client = None

    from remem.embedding_model import _get_embedding_client

    embedding_model = _get_embedding_client(
        global_config=config, embedding_model_name=args.embedding_name, openai_style_server=True
    )

    total_metrics = defaultdict(float)
    num_total_questions = 0
    # Global per-category aggregation across all samples (including cached ones)
    global_category_sums = defaultdict(lambda: defaultdict(float))  # cat -> metric -> sum over questions
    global_category_counts = defaultdict(int)  # cat -> #questions
    # Collect all serialized samples across conversations for final aggregate output
    aggregated_samples = []

    # Keep track of last inference_type encountered for naming aggregate output
    last_inference_type = None
    for sample_idx, file_path in tqdm(list(enumerate(files)), total=len(files)):
        try:
            sample = json.load(open(file_path))
        except Exception as e:
            logger.error("Failed to load %s: %s", file_path, e)
            continue

        # REALTALK: sessions and qa live at the root level
        conversation = sample
        qa_pairs = sample.get("qa", [])

        candidate_sessions = get_sessions(conversation)
        logger.info(
            "file %s (#sessions=%d, #qa=%d)", os.path.basename(file_path), len(candidate_sessions), len(qa_pairs)
        )

        dataset_name = f"{dataset_label}_{sample_idx}"
        config.dataset = dataset_name
        config.__post_init__()

        working_dir = get_working_dir(config.save_dir, dataset_name, llm_label, embedding_label, args.extract_method)
        inference_type = get_inference_type(args.agent_fixed_tools, args.agent_max_steps, args.extract_method)
        last_inference_type = inference_type
        rag_results_path = os.path.join(working_dir, f"rag_results_{inference_type}.json")

        # Skip if results exist
        if os.path.exists(rag_results_path) and not force_rag_from_scratch:
            logger.info("RAG results exist for %s, loading cached metrics...", dataset_name)
            try:
                with open(rag_results_path, "r") as f:
                    existing_results = json.load(f)
                    # Determine question_count from cached file for correct weighting; fall back safely
                    cached_num_questions = None
                    try:
                        cached_num_questions = (
                            int(existing_results.get("question_count"))
                            if existing_results.get("question_count") is not None
                            else None
                        )
                    except Exception:
                        cached_num_questions = None
                    if cached_num_questions is None:
                        # Fallback to number of serialized samples if available
                        samples_obj = existing_results.get("samples")
                        if isinstance(samples_obj, list):
                            cached_num_questions = len(samples_obj)
                    assert cached_num_questions is not None, "Failed to determine the cached number of questions"

                    if "overall_metrics" in existing_results and cached_num_questions is not None:
                        existing_metrics = existing_results["overall_metrics"]
                        for key in existing_metrics:
                            try:
                                total_metrics[key] += float(existing_metrics[key]) * cached_num_questions
                            except Exception:
                                pass
                        num_total_questions += cached_num_questions
                        logger.info(
                            "Loaded cached metrics for %s with question_count=%d", dataset_name, cached_num_questions
                        )
                    # Incorporate per-category metrics from cached results into global aggregations
                    if "categories" in existing_results and isinstance(existing_results["categories"], dict):
                        for cat, info in existing_results["categories"].items():
                            cnt = int(info.get("count", 0) or 0)
                            if cnt <= 0:
                                continue
                            global_category_counts[cat] += cnt
                            metrics_dict = info.get("metrics", {}) or {}
                            for mkey, mval in metrics_dict.items():
                                try:
                                    global_category_sums[cat][mkey] += float(mval) * cnt
                                except Exception:
                                    pass
                    # Append cached samples to aggregated_samples so final aggregate includes them
                    try:
                        cached_samples = existing_results.get("samples")
                        if isinstance(cached_samples, list):
                            for q_i, s in enumerate(cached_samples):
                                if isinstance(s, dict):
                                    s.setdefault("conversation_file", os.path.basename(file_path))
                                    s.setdefault("conversation_idx", sample_idx)
                                    s.setdefault("dataset_name", dataset_name)
                                    s.setdefault("question_idx", q_i)
                                    aggregated_samples.append(s)
                    except Exception as e_inner:
                        logger.warning("Failed to append cached samples for %s: %s", dataset_name, e_inner)
            except Exception as e:
                logger.error("Error loading existing results for %s: %s", dataset_name, e)
                logger.info("Will rerun this sample...")
            else:
                continue

        rag = ReMem(global_config=config, working_dir=working_dir, llm=llm_client)
        rag.set_embedding_model(embedding_model)

        rag.index(candidate_sessions)

        session_metrics = defaultdict(float)
        # For per-category aggregation
        session_category_metrics = defaultdict(lambda: defaultdict(float))
        session_category_counts = defaultdict(int)

        questions = []
        gold_docs = []
        gold_answers = []
        question_metadata = []

        for qa_idx, qa_pair in enumerate(qa_pairs):
            question = qa_pair.get("question", "")
            questions.append(question)
            cur_gold_answers = [str(qa_pair.get("answer", "no information available"))]
            gold_answers.append(cur_gold_answers)

            # normalize evidence strings potentially containing multiple refs
            new_evidence_list = []
            for evidence in qa_pair.get("evidence", []):
                ev = str(evidence)
                if "; " in ev:
                    new_evidence_list.extend([x.strip() for x in ev.split("; ")])
                elif ";" in ev:
                    new_evidence_list.extend([x.strip() for x in ev.split(";")])
                else:
                    new_evidence_list.append(ev)
            qa_pair["evidence"] = new_evidence_list

            cur_gold_docs = get_gold_docs_for_qa_pair(qa_pair, conversation)
            gold_docs.append(cur_gold_docs)

            question_metadata.append(
                {
                    "type": qa_pair.get("category"),
                }
            )

        selected_metrics = ("qa_em", "qa_f1", "qa_mem0_llm_judge", "qa_bleu1")
        qa_evaluators, retrieval_evaluators = rag.get_evaluators(gold_answers, gold_docs, selected_metrics)
        (query_solutions, all_response_message, all_metadata, session_retrieval_metrics, session_qa_metrics) = (
            rag.rag_for_qa(questions, gold_docs, gold_answers, selected_metrics, question_metadata=question_metadata)
        )

        num_total_questions += len(questions)

        for key in session_retrieval_metrics:
            total_metrics[key] += session_retrieval_metrics[key] * len(questions)
            session_metrics[key] += session_retrieval_metrics[key] * len(questions)
        for key in session_qa_metrics:
            total_metrics[key] += session_qa_metrics[key] * len(questions)
            session_metrics[key] += session_qa_metrics[key] * len(questions)

        # Aggregate per-category metrics from per-sample scores
        for idx, qs in enumerate(query_solutions):
            category_name = map_question_category(question_metadata[idx].get("type"))
            if not category_name:
                continue
            session_category_counts[category_name] += 1
            if qs.metrics:
                for metric_key, metric_val in qs.metrics.items():
                    try:
                        session_category_metrics[category_name][metric_key] += float(metric_val)
                    except Exception:
                        # Skip non-numeric values
                        pass

        # Update global per-category aggregations with this sample's raw sums
        for category, count in session_category_counts.items():
            global_category_counts[category] += count
            for mkey, total_val in session_category_metrics[category].items():
                global_category_sums[category][mkey] += total_val

        # log running averages
        for key in session_retrieval_metrics:
            logger.info("%s: %.4f", key, round(total_metrics[key] / max(1, num_total_questions), 4))
        for key in session_qa_metrics:
            logger.info("%s: %.4f", key, round(total_metrics[key] / max(1, num_total_questions), 4))

        # save results
        session_metrics = {key: round(session_metrics[key] / max(1, len(questions)), 4) for key in session_metrics}
        samples_dict = safe_serialize_query_solutions(query_solutions)
        rag_results = {"samples": samples_dict, "overall_metrics": session_metrics, "question_count": len(questions)}

        # Tag each sample with conversation metadata for later aggregation saving
        for local_q_idx, s in enumerate(samples_dict):
            try:
                s.setdefault("conversation_file", os.path.basename(file_path))
                s.setdefault("conversation_idx", sample_idx)
                s.setdefault("dataset_name", dataset_name)
                s.setdefault("question_idx", local_q_idx)
            except Exception:
                pass
        aggregated_samples.extend(samples_dict)

        # Add per-category summary
        if session_category_counts:
            category_summary = {}
            for category, count in session_category_counts.items():
                category_summary[category] = {"count": count, "metrics": {}}
                for metric_key, total_val in session_category_metrics[category].items():
                    category_summary[category]["metrics"][metric_key] = round(total_val / count, 4)
            rag_results["categories"] = category_summary

        rag_results_path = f"{rag.working_dir}/rag_results_{inference_type}.json"
        success = safe_dump_json(rag_results, rag_results_path)
        if not success:
            logger.warning(f"Warning: Had to use fallback serialization for {rag_results_path}")

    # Final summary prints: overall metrics and per-category metrics across all processed samples
    logger.info(f"Total QA pairs: {num_total_questions}")
    if num_total_questions > 0:
        logger.info("==== Overall Metrics (averaged over all questions) ====")
        for key in sorted(total_metrics.keys()):
            try:
                avg_val = total_metrics[key] / num_total_questions
                logger.info("%s: %.4f", key, round(avg_val, 4))
            except Exception:
                pass

    if global_category_counts:
        logger.info("==== Metrics by Question Category ====")
        for cat in sorted(global_category_counts.keys()):
            cnt = global_category_counts[cat]
            logger.info("Category '%s' (n=%d)", cat, cnt)
            if cnt <= 0:
                continue
            metrics_for_cat = global_category_sums[cat]
            for mkey in sorted(metrics_for_cat.keys()):
                try:
                    avg_val = metrics_for_cat[mkey] / cnt
                    logger.info("  %s: %.4f", mkey, round(avg_val, 4))
                except Exception:
                    pass

    # Save aggregate results across all conversations
    try:
        if num_total_questions > 0 and aggregated_samples:
            # Deterministic ordering
            try:
                aggregated_samples.sort(key=lambda x: (x.get("conversation_idx", 10**9), x.get("question_idx", 10**9)))
            except Exception:
                pass
            overall_metrics = {}
            for key, total in total_metrics.items():
                try:
                    overall_metrics[key] = round(total / num_total_questions, 4)
                except Exception:
                    pass

            # Category summary
            category_summary = {}
            for cat, cnt in global_category_counts.items():
                if cnt <= 0:
                    continue
                cat_metrics = {}
                for mkey, total_val in global_category_sums[cat].items():
                    try:
                        cat_metrics[mkey] = round(total_val / cnt, 4)
                    except Exception:
                        pass
                category_summary[cat] = {"count": cnt, "metrics": cat_metrics}

            # Derive inference suffix similar to other overall eval scripts
            graph_type = config.graph_type if hasattr(config, "graph_type") else "graph"
            inference_suffix = ""
            if graph_type != "dpr_only" and last_inference_type:
                inference_suffix = "_" + last_inference_type.replace("/", "_")

            os.makedirs("outputs/realtalk", exist_ok=True)
            aggregate_path = (
                f"outputs/realtalk/rag_results_{num_total_questions}_{llm_label}_{graph_type}{inference_suffix}.json"
            )

            aggregate_json = {
                "num_samples": len(aggregated_samples),  # number of QA samples aggregated
                "question_count": num_total_questions,  # total questions processed
                "overall_metrics": overall_metrics,
                "samples": aggregated_samples,
            }
            if category_summary:
                aggregate_json["categories"] = category_summary

            # Agent session statistics
            agent_session_stats = {
                "total_samples_with_logs": 0,
                "total_steps": 0,
                "avg_steps_per_query": 0,
                "step_distribution": defaultdict(int),
                "max_steps_distribution": defaultdict(int),
            }
            for s in aggregated_samples:
                logs = s.get("agent_session_logs") if isinstance(s, dict) else None
                if not logs:
                    continue
                agent_session_stats["total_samples_with_logs"] += 1
                steps = logs.get("num_steps", 0)
                agent_session_stats["total_steps"] += steps
                agent_session_stats["step_distribution"][steps] += 1
                max_steps = logs.get("max_steps", 0)
                agent_session_stats["max_steps_distribution"][max_steps] += 1

            if agent_session_stats["total_samples_with_logs"] > 0:
                agent_session_stats["avg_steps_per_query"] = round(
                    agent_session_stats["total_steps"] / agent_session_stats["total_samples_with_logs"], 2
                )
                aggregate_json["agent_session_stats"] = {
                    "total_samples_with_logs": agent_session_stats["total_samples_with_logs"],
                    "total_steps": agent_session_stats["total_steps"],
                    "avg_steps_per_query": agent_session_stats["avg_steps_per_query"],
                    "step_distribution": dict(agent_session_stats["step_distribution"]),
                    "max_steps_distribution": dict(agent_session_stats["max_steps_distribution"]),
                }

            # Raw question type summary (unmapped numeric/string type field)
            raw_type_metrics = defaultdict(lambda: defaultdict(float))
            raw_type_counts = defaultdict(int)
            for s in aggregated_samples:
                qm = s.get("question_metadata") if isinstance(s, dict) else None
                sm = s.get("sample_metrics") if isinstance(s, dict) else None
                if not qm or not sm:
                    continue
                qtype_raw = qm.get("type")
                if qtype_raw is None:
                    continue
                raw_type_counts[qtype_raw] += 1
                for mkey, mval in sm.items():
                    try:
                        raw_type_metrics[qtype_raw][mkey] += float(mval)
                    except Exception:
                        pass
            if raw_type_counts:
                aggregate_json["question_type_summary"] = {
                    str(qt): {
                        "count": raw_type_counts[qt],
                        "metrics": {mk: round(tv / raw_type_counts[qt], 4) for mk, tv in raw_type_metrics[qt].items()},
                    }
                    for qt in raw_type_counts
                }

            # Derive retrieval / QA error subsets if possible
            retrieval_error_samples = []
            qa_error_samples = []
            for s in aggregated_samples:
                sm = s.get("sample_metrics", {}) or {}
                # retrieval error if any recall-like metric < 1
                if any(
                    (k.lower().startswith("recall") and isinstance(v, (int, float)) and v < 1.0) for k, v in sm.items()
                ):
                    retrieval_error_samples.append(s)
                # qa error if exact match metric present and < 1
                em_val = sm.get("qa_em")
                if isinstance(em_val, (int, float)) and em_val < 1.0:
                    qa_error_samples.append(s)

            with open(aggregate_path, "w") as f:
                json.dump(aggregate_json, f, indent=4)
            logger.info("Saved REALTALK results to %s", aggregate_path)

            # Save error files (best-effort)
            try:
                if retrieval_error_samples:
                    with open(aggregate_path.replace("rag_results", "retrieval_error"), "w") as f:
                        json.dump(retrieval_error_samples, f, indent=4)
                if qa_error_samples:
                    with open(aggregate_path.replace("rag_results", "qa_error"), "w") as f:
                        json.dump(qa_error_samples, f, indent=4)
            except Exception as e:
                logger.warning("Failed to save error subsets: %s", e)
    except Exception as e:
        logger.error("Failed to save aggregate REALTALK results: %s", e)
