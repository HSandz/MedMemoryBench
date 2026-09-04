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
claims; historical versions are excluded before classification, so a new claim
with no active/contested compatible target is recorded as `NEW`. Evidence
references and claim/episode graph edges are maintained through one store
operation.

LLM-proposed subject IDs are checked against source speakers and the visible
conversation scope; ambiguous multi-speaker self-references are discarded.
State claims carry a value-independent `state_slot` (for example,
`residence_location`) as internal compiler metadata. It is kept out of claim
retrieval text and final answer context; slot-only naturalized labels are used
for compiler matching. Only active or contested same-subject claims with an
exact slot match or slot cosine similarity at least
`state_candidate_min_similarity` are sent to the update classifier.
Superseded/refined versions remain history, not mutation targets.

Canonical subjects are derived from visible scope, participants, and cited turns.
In primary-user scope, unrecognized attribute-like subjects resolve to
`primary_user`; `speaker:<name>` is emitted only for a visible participant.
Generic consultations remain `general_non_personal`, and third-party scopes stay
isolated. Claims must cite valid source turn IDs; model-facing wrappers are
resolved only when they map deterministically to an existing visible ID (exact
matches always win). Independently valid claims and the episode archive are
retained when another claim is malformed or ungrounded.
Grounding-only repair must preserve the claim's normalized semantic fingerprint
(subject, subject ID, predicate, value, qualifiers, polarity, modality,
persistence, state slot, and valid-time fields); only provenance may change.
Candidates that alter this fingerprint are rejected and counted separately from
ordinary unresolved grounding failures. Schema-invalid claims retain the bounded
conservative repair path but cannot introduce unrelated claims.
Turn IDs are canonical strings throughout prompts, validation, and persisted
evidence (for example, numeric `0` becomes `"0"`); equivalent JSON
representations do not trigger repair. Duplicate canonical IDs within a session
are deterministically namespaced. Build telemetry exposes allowed IDs, capped
invalid-ID samples, normalized presentation-reference counts, claim-level
grounding failures, preserved-valid-claim counts, repair outcomes, and claim-limit
overflow counts.
The configured `max_claims_per_episode` is applied before claim validation;
claims beyond the limit are ignored in source order and counted only as excess.
Setting it to `0` retains the episode while accepting no state claims.
When non-identical claims share an episode, classification receives only the
exact cited turns for the new and candidate claims (bounded to four turns per
side), so unrelated dialogue cannot affect correction/restatement decisions.
Exact normalized duplicates remain deterministic.

### LLM prompt contracts

All Event-State core LLM prompts treat conversation turns, stored or retrieved
memory, candidate claims, and prior model output as data rather than
instructions. This boundary applies during extraction, state classification,
optional retrieval planning, structured-output repair, and final answering;
instructions embedded in evidence do not override the active task.

Extraction requests a concise, source-grounded `episode_summary` of salient
events, observations, decisions, commitments, and context rather than a claim
list. Predicates are concise, value-independent relation/property labels.
`valid_time_text` preserves source wording, while `valid_from` and `valid_to`
are populated only for explicit, well-supported absolute bounds; unknown time
remains unknown. Extractor confidence is stored as the neutral compatibility
sentinel `0.5` and does not affect Event-State semantic decisions.

Structural and claim-subset repair use a dedicated repair-only system prompt;
they may correct permitted structure or provenance but cannot perform fresh
extraction or classification. The update classifier may select only `NEW`,
`DUPLICATE`, `CORROBORATE`, `REFINE`, `SUPERSEDE`, or `CONFLICT`. `EPISODIC`
remains a deterministic internal outcome for non-observation claims. Classifier
confidence measures support for the proposed relation from supplied evidence,
and its rationale is a short audit reason.

Extraction lifecycle labels are semantic: `state` is a currently true or
ongoing proposition worth carrying forward (even if it began long ago),
`history` is former or completed background no longer asserted as current, and
`episode` is a bounded occurrence, action, recommendation, measurement, or
other transient event. Assistant generic explanations and paraphrases are not
stored as user facts; explicit conversation-specific instructions,
recommendations, decisions, corrections, or status assertions may be retained.
Build telemetry reports extracted counts for each lifecycle and separately
reports final active, superseded, refined, contested, standalone-history, and
standalone-episode counts.

Within one episode, `CORROBORATE` is deterministically downgraded to
`DUPLICATE`. `SUPERSEDE`, `REFINE`, and `CONFLICT` require an explicit
same-episode evidence relation and source-turn ordering from the episode's
chronological turn evidence. Invalid transitions preserve the old state and
fall back to `DUPLICATE` for explicit restatements or `NEW` otherwise.

Direct claim retrieval exposes active/contested state representatives and
standalone historical claims. Superseded and refined state versions remain in
the store and are rendered as a bounded, predecessor-first `Prior states` section
when their current representative is shown; they do not independently compete
in the current-state claim pool.

Extraction requests JSON-object mode when the configured client supports it. If
the complete envelope cannot be parsed, Event-State first scans the literal
response for complete JSON claim objects using quote-, escape-, and nesting-aware
balancing, then makes one JSON/structure-only repair request. If the repair also
fails and no salvaged claim passed normal validation, it makes one final extraction
from the original visible conversation. The sequence is bounded to three LLM
calls. Unsupported JSON mode retries the same call without `response_format` and
retains the configured extraction temperature and token limit. Provider failures
are not treated as malformed JSON.

If every semantic extraction attempt fails, the immutable episode, raw text, and
turn evidence are still stored. Its summary becomes a bounded chronological
rendering of all visible turns with speaker and turn labels. Each turn receives a
share of the budget and long turns preserve both their beginning and ending text,
so late-session information is not silently discarded. Build telemetry separates
fragment salvage, structural repair, recovery extraction, and semantic-unavailable
counts; malformed outputs also retain bounded previews and SHA-256 diagnostics.

Snapshots use schema version 4 plus the Event-State build semantic version.
Semantic version `2.9` adds extraction recovery and temporal-ingress validation
to the `2.8` non-exact state-value relation semantics: equivalent observations
corroborate an active claim across sessions (or are
duplicates in the same episode), refinements become more specific current
representations, material changes supersede, contradictions remain contested,
and uncertain decisions create a new claim. New `state` observations enter the
compiler with an open `valid_to`; an extractor-provided closing bound is cleared.
For observed/asserted state, a safely comparable `valid_from` later than the
record time is cleared. Closed historical intervals and future planned episodic
dates remain valid, while any safely comparable reversed interval is cleared.
Temporal wording is retained in `valid_time_text`; no replacement date is
invented. Build telemetry reports ingress guard, future-state-start, state-end,
and invalid-interval counts. The classifier must provide
`state_value_relation` as one of `equivalent`, `refinement`, `changed`,
`contradictory`, or `uncertain`; incompatible operation/relation pairs safely
fall back to `NEW`. `2.5`, `2.6`, `2.7`, and `2.8` snapshots use older build
semantics and must be rebuilt. Version 1 and 2 snapshots are rejected rather
than silently reinterpreted. Claims retain canonical subject IDs and lifecycle
statuses (`active`, `superseded`, `refined`, `contested`, or `standalone`)
together with their evidence references. Dataset adapters may provide a
canonical `recorded_at` alongside an immutable `recorded_at_raw` display value.
Event-State uses the canonical record time for its existing generic temporal
helpers and preserves the raw value in episode metadata. This does not infer
claim valid time or introduce dataset-specific retrieval behavior.

## Configuration

Start with `configs/method_config/event_state_gemini.yaml` (or the
`event_state_gpt-5.1.yaml` OpenAI variant). Build settings live under
`build_config`; retrieval-only ablations live under `retrieval_config`.
`memorize_model` controls extraction and state classification, while `model`
controls final answers. The embedding backend supports repository-compatible
local/HuggingFace and OpenAI configurations.

### Optional adaptive retrieval planner

The query-only planner is disabled by default. Under `retrieval_config`, use:

```yaml
planner_rounds: 0       # preserves the ordinary one-call query flow
planner_max_requests: 3
planner_temperature: 0.0
planner_max_tokens: 1200
planner_merge_mode: coverage_interleave
```

When enabled, the answer model chooses either a final answer or a bounded JSON
retrieval plan. The controller answers only when current personalized evidence
explicitly supports every personalized premise; missing, ambiguous,
contradicted, or temporally incompatible premises trigger retrieval. General
knowledge may support reasoning after those premises are present, but never
substitutes for personal memory. Valid plans execute locally with the existing
claim/episode retrieval, RRF, MMR, provenance, and context limits. Planner
requests with explicit or safely derivable dates carry normalized structured
time constraints; the planner path does not invoke the lexical temporal parser.
At most
`planner_rounds + 1` query-model calls are made. Planner mode does not use the
lexical temporal cue parser; temporal requests must contain validated
structured dates. Structured planner filtering supports only record-axis modes:
`record_exact`, `record_before`, `record_after`, and `record_interval`.
`knowledge_as_of` with `state_view: as_of` retains historical knowledge
semantics. ESHM continues to store bitemporal claim state (`t_record` and
`t_valid`), including valid intervals used for state versioning and
knowledge-as-of interpretation. It does not expose arbitrary valid/event-time
filters to the planner because episodes do not carry universally authoritative
event time. Event-date questions without an explicit record-date axis remain
semantic retrieval requests with no hard temporal filter; unknown time is not
treated as weak temporal evidence. Older experimental artifacts may retain
`valid_*` request data as diagnostics, but those modes are rejected for new
planner output and are never remapped to record-time modes. Invalid or duplicate
requests are skipped and malformed planner output falls back to one ordinary
final-answer call. Planner mode is realtime-only because dependent rounds cannot
use the existing single-stage batch-answer transport; `planner_rounds: 0`
retains batch support unchanged. Each planner prompt reports retrieval rounds
available including the current decision, plus any additional rounds afterward,
so a decision with one configured round is explicitly allowed to retrieve.

Planner-enabled outer fusion uses `coverage_interleave` by default: relevance
for a memory is the maximum reciprocal rank contributed by any channel, while
channel support is diagnostic only. The bounded candidate pool is built by
deterministically interleaving rank 1 from the base channel and each planner
request, then rank 2 from each channel, and so on; the existing state-aware MMR
selector remains the final evidence selector and consumes the merged relevance
normalized across that final pool. `planner_merge_mode: sum_rrf` is available
only to reproduce the historical agreement-weighted behavior. Retrieval records
expose `planner_channel_retrieval` when an item appeared in any planner request
channel and `planner_added_to_final` when it was absent from the base selected
evidence and newly selected after planner fusion; the legacy
`planner_retrieval` field remains an alias for channel participation.
Planner diagnostics are persisted per query in both the query-answer artifact
and its retrieval-record sidecar. Planner failures distinguish
`json_parse_failure_count` from `schema_validation_failure_count`; the legacy
`parse_failure_count` is their aggregate. Bounded failure diagnostics include
the stage, a validator/decoder reason, SHA-256, and a maximum 500-character
preview, never hidden reasoning. Planner artifacts also include a bounded
`candidate_trace` with `base_ranked`, each executed `planner_channels` entry,
and `merged_preselect`; entries contain IDs, type, rank, scores, temporal
metadata, and sorted source-session provenance only. This is diagnostic
telemetry and does not add retrieval logic. Query usage reports expose
`event_state.plan_or_answer` as `planner_controller` and
`event_state.final_answer` alongside ordinary answer operations.

The default `state_current_candidate_top_k` is `3`; lower it only when the
classifier prompt budget requires fewer current-state candidates.

Temporal retrieval is query-only and is enabled by `temporal_retrieval_enabled`
(`true` by default), with a neutral `temporal_retrieval_weight` of `1.0`.
The deterministic parser recognizes explicit `YYYY-MM-DD` and `YYYY/MM/DD`
dates with `as of`/`by` (state as-of), `before`, `after`, `between ... and ...`
or `from ... to ...` (record-time bounds), and record cues such as `record
dated DATE` (exact record time). Planner valid-time constraints consume claim
`valid_from`/`valid_to` metadata and never reinterpret an episode's
`recorded_at` as event time. Episodes remain available through the ordinary
dense channel when valid-time metadata is absent. A date without a clear axis uses a hybrid
record/valid-time candidate channel; incomplete expressions such as "around the
5th" are ignored and use ordinary retrieval.

Episode temporal candidates require a parseable `recorded_at` satisfying the
bound (for as-of, recorded on or before the target). Claims use actual EvidenceRefs to matching episodes and, for as-of
queries, their structured valid interval as a secondary check. As-of retrieval
uses knowledge-as-of semantics: state evidence recorded after the target is
excluded, and known valid intervals use `[valid_from, valid_to)` (inclusive
start, exclusive end). Claims without valid bounds may use a lower-confidence
record-time fallback but no interval is fabricated. Temporal candidates are
added to the existing claim/episode RRF channels and still pass the unchanged
candidate limit and state-aware MMR selector. Query diagnostics expose the
constraint, candidate counts, future-state filtering, and temporal contribution
on selected records; querying never mutates a snapshot.

Answer judging is independent from query execution. If a response and
retrieval complete but the later judge fails, the query remains in the query
answer and retrieval-record artifacts with `evaluation_status: "judge_failed"`,
`score: null`, `is_correct: null`, model output, retrieval quality, planner
diagnostics, execution usage, and bounded judge-failure metadata. Answer
metrics exclude it; retrieval metrics include it when retrieval ground truth is
available. Resuming deferred judges is idempotent. This query-only work does
not change the write-side semantic version, which remains `2.9`.

`max_episode_source_excerpts_total` is an optional query-only
`retrieval_config` setting with a default of `2`. It selects one global set of
raw source turns across all selected episodes, rather than two turns per
episode. Claim-provenance excerpts have priority: their exact
`(episode_id, source_turn_id)` pairs are excluded before generic episode turns
are embedded and scored against the query. Ties are deterministic by selected
episode rank, source-turn position, episode ID, and turn ID; surviving excerpts
render in selected-memory and source-conversation order. This setting changes
query artifacts but not memory build compatibility or the serialized snapshot.

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

# Build a memory snapshot, then answer from it (MedMemoryBench or LoCoMo)
python main.py -m event_state_gemini -d medmemorybench --stage memory
python main.py -m event_state_gemini -d locomo --stage memory
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
The evaluator shows a `Memory build` progress bar covering both preparation and
ordered stateful commit steps; it reaches completion only after commits finish.
For LoCoMo, Event-State bypasses character chunking: every chronological session
in one sample is prepared as a source session, then committed in sample order.
Its sample-global `source_session_index`, original timestamps, session IDs, and
turn IDs are persisted. Event-State snapshots are written per LoCoMo sample;
`--stage query` restores those exact stores without rebuilding memory. Query
workers use isolated restored stores, and batch preparation likewise freezes
each sample's retrieval result before the run-wide `query-final` batch stage.
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
When enabled, claim source expansion scores every immutable turn cited by every
claim EvidenceRef with the query embedding, ranks each reference by its best
cited-turn cosine score, and keeps at most `max_source_excerpts_per_claim`
references with at most three turns each. Selected turns are rendered back in
immutable source order. The same selected turn keys drive claim blocks,
claim-vs-episode deduplication, and `included_provenance_evidence`; lineage
metadata therefore reports only supporting blocks actually included in the
answer context. Legacy EvidenceRefs whose cited IDs cannot be resolved retain
the deterministic first-turn fallback. Generic episode source evidence remains
under the separate global budget, and `selected_episode_evidence_excerpt_count`
is its total excerpt count for the query; candidate and claim-dedup counts are
also reported. Build telemetry reports state-value relation counts, the relation
consistency guard count, and cross-session corroboration counts alongside
multi-reference and multi-session provenance totals.

The evaluator keeps the legacy session metric for compatibility and adds
Event-State `claim_lineage` (all evidence attached to selected claims) and
`answer_visible` diagnostics. `answer_visible` includes selected episodes only
when their full block fits the final prompt. A state claim contributes its
direct origin only when its compact state block fits; otherwise, it contributes
only included provenance excerpts, and is excluded entirely when neither is
visible.

### MedMemoryBench mixed-source integrity

MedMemoryBench assigns every loaded conversation a deterministic opaque source
UID (for example, `src_p1_r17`). This is the only session identity passed to a
memory method and is stored in Event-State episode and retrieval provenance.
The dataset adapter also supplies only generic `conversation_scope` metadata:
`primary_user`, `general_non_personal`, or `third_party:<visible identity>`.
It does not pass a noise label, noise type, or benchmark gold ID to Event-State.

Original clean benchmark session IDs remain evaluator-private. During retrieval
scoring, the evaluator translates selected source UIDs through its private
source-to-benchmark mapping; distractor UIDs map to null and therefore cannot
be gold evidence. Query diagnostics report selected clean/distractor source
counts and noise-intrusion rates without exposing that telemetry to the answer
model. Mixed runs fail before memory construction on duplicate source UIDs or
invalid clean/distractor provenance. Event-State also rejects duplicate source
UIDs in its immutable episode archive and reports
`duplicate_episode_source_id_count` (normally zero).

Mixed artifacts with the former positional-ID collision are historical and
must not be used for robustness comparisons. New snapshots include
`source_identity_schema_version: 1`; their legacy-shaped `session_ids` field,
where present, contains method-facing source UIDs rather than gold session IDs.
New `medmemorybench.query_answers` artifacts use version 3; version-2 artifacts
retain their historical, ambiguous source-session semantics and are not
reinterpreted during loading.

`SUPERSEDE` interval closure is conservative. Ordinary state changes use the
new claim's normalized `valid_from` only when it is not earlier than the
predecessor's known `valid_from` or `recorded_at`; otherwise the compiler uses
the new claim's `recorded_at` as a record-time fallback, or leaves the prior
interval open when no safe date is available. Only an explicit classifier
relation of `correction` may apply a retroactive `valid_from`, and only when it
does not make `valid_to` precede the predecessor's `valid_from`. Build
telemetry reports `supersede_temporal_guard_count`,
`supersede_record_time_fallback_count`, and
`retroactive_correction_applied_count`.
