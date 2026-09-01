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
matches always win). Malformed or ungrounded claims are repaired once as an
invalid subset and then dropped while independently valid claims and the episode
archive are retained.
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
grounding failures, preserved-valid-claim counts, subset-repair outcomes, and
claim-limit overflow counts. Subset repair reports successes for partial recovery
and failures for unresolved requested claims; a repair failure does not imply that
retained valid claims are unusable.
The configured `max_claims_per_episode` is applied before claim validation;
claims beyond the limit are ignored in source order and counted only as excess.
Setting it to `0` retains the episode while accepting no state claims.
When non-identical claims share an episode, classification receives only the
exact cited turns for the new and candidate claims (bounded to four turns per
side), so unrelated dialogue cannot affect correction/restatement decisions.
Exact normalized duplicates remain deterministic.

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

If extraction and its single repair attempt both fail, the retained episode
summary is a bounded chronological rendering of all visible turns with speaker
and turn labels. Each turn receives a share of the budget and long turns
preserve both their beginning and ending text, so late-session information is
not silently discarded.

Snapshots use schema version 4 plus the Event-State build semantic version.
Semantic version `2.8` adds explicit non-exact state-value relation semantics:
equivalent observations corroborate an active claim across sessions (or are
duplicates in the same episode), refinements become more specific current
representations, material changes supersede, contradictions remain contested,
and uncertain decisions create a new claim. The classifier must provide
`state_value_relation` as one of `equivalent`, `refinement`, `changed`,
`contradictory`, or `uncertain`; incompatible operation/relation pairs safely
fall back to `NEW`. `2.5`, `2.6`, and `2.7` snapshots use older build semantics
and must be rebuilt. Version 1 and 2 snapshots are rejected rather than
silently reinterpreted. Claims retain canonical subject IDs and lifecycle statuses (`active`,
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
The evaluator shows a `Memory build` progress bar covering both preparation and
ordered stateful commit steps; it reaches completion only after commits finish.
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
excerpts, prefers referenced turn IDs, and supplements them with the global
episode source set. `selected_episode_evidence_excerpt_count` is the total
generic excerpt count for the query; candidate and claim-dedup counts are also
reported. Build telemetry reports state-value relation counts, the relation
consistency guard count, and cross-session corroboration counts alongside
multi-reference and multi-session provenance totals.

The evaluator keeps the legacy session metric for compatibility and adds
Event-State `claim_lineage` (all evidence attached to selected claims) and
`answer_visible` diagnostics. `answer_visible` includes selected episodes only
when their full block fits the final prompt. A state claim contributes its
direct origin only when its compact state block fits; otherwise, it contributes
only included provenance excerpts, and is excluded entirely when neither is
visible.

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
