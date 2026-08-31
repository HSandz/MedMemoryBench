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

Canonical subjects are derived from visible scope, participants, and cited turns.
In primary-user scope, unrecognized attribute-like subjects resolve to
`primary_user`; `speaker:<name>` is emitted only for a visible participant.
Generic consultations remain `general_non_personal`, and third-party scopes stay
isolated. Claims must cite valid source turn IDs; malformed or ungrounded claims
are repaired once and then dropped while the episode archive is retained.
When non-identical claims share an episode, classification receives only the
exact cited turns for the new and candidate claims (bounded to four turns per
side), so unrelated dialogue cannot affect correction/restatement decisions.
Exact normalized duplicates remain deterministic.

Snapshots use schema version 3 plus the Event-State build semantic version. The
corrected builder is semantic version `2.0`; snapshots built without that version
do not match the build compatibility hash and must be rebuilt. Version 1 and 2
snapshots are rejected rather than silently reinterpreted. Claims retain canonical
subject IDs and lifecycle statuses (`active`,
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

Structured memory construction uses bounded completion budgets (3,200 extraction
tokens and 1,600 update-classification tokens in the Persona-1 configuration) and
disables model thinking for those JSON-only calls.

The Persona-1 NIM comparison configuration keeps thinking enabled only for final
answer generation (`reasoning_budget: 5000`); its `memorize_model` disables
thinking. The Persona-1 Gemini configuration uses `gemini-2.5-flash` for both
answering and memory construction.

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

### Parallel memory construction

Use the existing global worker option, for example `--workers 4`, with Event-State.
Each source session performs extraction, repair/validation, episode construction,
subject resolution, and episode/claim embedding in a bounded preparation pool.
Conversation scope is resolved independently from each source session's turns (or
its explicit scope metadata), so mixed primary-user, third-party, and general
sessions remain isolated during staging.
Prepared sessions are then committed to one store strictly in original session
order; candidate generation, update classification, and every state/provenance
mutation remain serial. `--workers 1` follows the same staged path without a
pool, and workers greater than one are intended to be semantically equivalent.
LoCoMo multi-session chunks use the same ordered preparation/commit split, while
query workers continue to use the evaluator's existing global query limit.
MedMemoryBench unit build telemetry reports wall-clock preparation plus ordered
commit time (parallel worker durations are not summed).
Worker count is an execution setting and is not included in snapshot or config
identity hashes.

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
