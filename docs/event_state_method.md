# Event-State Hybrid Memory

`event_state` is a standalone `BaseAgent` implementation that keeps immutable
conversation episodes alongside compact, versioned semantic claims. Repeated
claims share evidence references; later changes create superseded versions and
unresolved contradictions remain contested. Retrieval independently searches
claims and episode summaries, fuses ranks, and selects a small top-k/MMR
evidence set. Optional typed PPR is query-only and does not alter snapshots.

Claim self-references are resolved from cited source turns, so alternating
named speakers remain separate subject namespaces. Repeating an older
superseded value creates a new version and never reactivates the historical
node. `SUPERSEDE`, `REFINE`, and `CONFLICT` may target only active or contested
claims; an LLM attempt to transition historical state is recorded as a new
claim with the `historical_transition_target` fallback reason. Evidence
references and claim/episode graph edges are maintained through one store
operation.

LLM-proposed subject IDs are checked against source speakers and the visible
conversation scope; ambiguous multi-speaker self-references are discarded.
State compilation always reserves up to three active or contested same-subject
claims for classification, even when their embedding similarity is below the
historical candidate threshold.

Snapshots use schema version 3. Version 1 and 2 snapshots are rejected rather than
silently reinterpreted; rebuild the memory when upgrading from the prototype
format. Claims retain canonical subject IDs and lifecycle statuses (`active`,
`superseded`, `refined`, `contested`, or `standalone`) together with their
evidence references.

## Configuration

Start with `configs/method_config/event_state_gemini.yaml` (or the
`event_state_gpt-5.1.yaml` OpenAI variant). Build settings live under
`build_config`; retrieval-only ablations live under `retrieval_config`.
`memorize_model` controls extraction and state classification, while `model`
controls final answers. The embedding backend supports repository-compatible
local/HuggingFace and OpenAI configurations.

The default `state_current_candidate_top_k` is `3`; lower it only when the
classifier prompt budget requires fewer current-state candidates.

## Commands

```bash
# Validate registration without calling providers
python main.py --list-methods
python main.py -m event_state_gemini -d medmemorybench --dry-run

# Small evaluation runs (set dataset limits in the dataset config)
python main.py -m event_state_gemini -d medmemorybench
python main.py -m event_state_gemini -d locomo

# Build a memory snapshot, then answer from it
python main.py -m event_state_gemini -d medmemorybench --stage memory
python main.py --stage query --memory-run <memory-run-directory>
```

To compare PPR policies on the same build, use a query-stage YAML override
with `retrieval_config.ppr_enabled: false` and then `true`; no extraction or
embedding calls are required after the snapshot is imported. No benchmark
annotations are passed to the method during memory construction or retrieval.

Query diagnostics keep dense, fusion, PPR, final, and selection scores
separate, and report `selected_ids` independently from `included_ids`. State
claims expose `all_provenance_evidence` for complete lineage and
`included_provenance_evidence` for evidence that survived context budgeting.
When enabled, source expansion follows claim references to compact turn
excerpts, prefers referenced turn IDs, and deduplicates episodes already in
the selected context.

The evaluator keeps the legacy session metric for compatibility and adds
Event-State `claim_lineage` (all evidence attached to selected claims) and
`answer_visible` diagnostics. `answer_visible` includes selected episodes only
when their full block fits the final prompt. A state claim contributes its
direct origin only when its compact state block fits; otherwise, it contributes
only included provenance excerpts, and is excluded entirely when neither is
visible.
