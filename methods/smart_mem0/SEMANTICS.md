# SmartMem0 Semantics

## Constitution: General-Domain First

Architectural decisions must be domain-neutral, benchmark-neutral and language-neutral
at the control level. Domain aliases, English stemming and benchmark wrappers may not
authorize eligibility, terminal output, state merging or context expansion.

## Durable Records

MemoryAtom contains a stored assertion, owner, predicate, object, value, stance,
modality, temporal axes and linked EvidenceRecords. Source speaker is not fact owner.
StateIdentity is the normalized tuple (owner, scope, predicate, object). Values,
scope_entities and times are not identity. Missing owner/predicate produces no state
identity; the shared resolver does not fabricate patient or primary_user ownership.
Namespaces and predicate suffixes are preserved rather than guessed equivalent.

Typed relations remain in the ledger. SUPPORT/RELATED never prove answers or expand
READ. SUPERSEDE/REFINE/CONFLICT retain state semantics. CAUSES requires endpoint and
relation evidence; time order alone is not causality.

## Query Contracts

QuestionInput has text, optional candidates, optional explicit owner_id and optional
selector_required. Raw strings use structural enumeration only, without language cue
lists. Projection is ENTITY, VALUE, DATE, TEXT or OPTION_SET. DATE is retained for
compatibility, not renamed to imply duration support that does not exist.

Advisor output is a proposal, not a certificate. Exact stored predicate binding is
required for terminal support; lexical rarity only measures relevance. Unknown
paraphrases must go to synthesis. Multiple focus spans do not veto an atomic answer.
The certificate is not a general semantic theorem prover: exact predicates and
structured selectors cannot guarantee interpretation of every free-text qualifier.

Selectors respect event_time, document_time, origin_document_time or explicitly
requested effective_event_time. No implicit axis substitution. An invalid selector
preserves eligible evidence but blocks terminal when required or unspecified. Caller
selector_required=False ignores the selector; True requires a valid resolved selector.

## Boundaries

BaseWorld is a subset of CandidateWorld, whose size is at most 16. Certificate is
read-only and never retrieves. Final context IDs are a subset of CandidateWorld.
Evidence is opened only through selected memories' pointers. Context reserves supports,
competitors, hint premises, option associations and requested stored causal endpoints;
there is no unconditional fill-to-eight tail. Associations are not verdicts.

One advisor call is mandatory; the answer call is conditional. No repair/planner/gate
calls are allowed between them. Counts and token usage record the actual lifecycle.

## Scope of This Revision

Capture/consolidation and their historical prompt are frozen. Shared normalization
changes require schema 11. Full WRITE language/owner generalization remains explicit
follow-up work, not a claim that old memories are already universal.
