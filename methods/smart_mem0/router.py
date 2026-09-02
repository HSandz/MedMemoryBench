import re
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class RouteDecision:
    route: str  # EXACT, STATE_LATEST, TEMPORAL, HARD
    subject_id: str = "primary_user"
    target_concept: str = ""
    answer_slot: str = "VALUE"  # VALUE, DATE
    temporal_axis: str = "event_time"  # event_time, document_time
    temporal_relation: str = "MATCH"  # EARLIEST, LATEST, MATCH
    temporal_anchor: str = ""
    temporal_precision: str = ""  # YEAR, MONTH, DAY

class DeterministicRouter:
    """
    100% deterministic (zero-LLM) query router for SmartMem0 P1A.
    Fails open to HARD if there is any ambiguity.
    """
    
    # Generic subject regexes
    RE_PRIMARY = re.compile(r'\b(my|i|patient|user|me)\b', re.IGNORECASE)
    RE_THIRD_PARTY = re.compile(r'\bmy\s+([a-z]+)\b', re.IGNORECASE)
    
    # HARD triggers
    RE_HARD = re.compile(
        r'\b(compare|compared|why|how|should|options|or|cause|because|result in)\b|\ba\s*,\s*b\s*,\s*c\b',
        re.IGNORECASE
    )
    
    # TEMPORAL triggers
    RE_TEMPORAL_DATE = re.compile(r'\b(when|what date|what time)\b', re.IGNORECASE)
    RE_TEMPORAL_ANCHOR = re.compile(r'\b(?:on|in|during|at)\s+(?:the\s+)?((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|[a-z]+\s+\d{4})\b', re.IGNORECASE)
    
    # EVENT vs DOC time triggers
    RE_DOC_TIME = re.compile(r'\b(documented|recorded|reported|mentioned|visit)\b', re.IGNORECASE)
    RE_EVENT_TIME = re.compile(r'\b(started|began|first appeared|onset|occurred|taking|took)\b', re.IGNORECASE)
    
    # RELATIONS
    RE_EARLIEST = re.compile(r'\b(first|earliest|initial|started|start)\b', re.IGNORECASE)
    RE_LATEST = re.compile(r'\b(latest|last|most recent)\b', re.IGNORECASE)
    
    # STATE_LATEST triggers
    RE_STATE_LATEST = re.compile(r'\b(current|latest|now)\b', re.IGNORECASE)
    
    def __init__(self, subject_postings: dict = None):
        # subject_postings maps subject_id -> set(memory_ids)
        self.subject_postings = subject_postings or {}

    def route_query(self, question: str) -> RouteDecision:
        if self.RE_HARD.search(question):
            return RouteDecision(route="HARD")
            
        subject_id = self._resolve_subject(question)
        if subject_id == "HARD":
            return RouteDecision(route="HARD")
            
        # 2. TEMPORAL (DATE or VALUE-at-time)
        temporal_match = self.RE_TEMPORAL_DATE.search(question)
        anchor_match = self.RE_TEMPORAL_ANCHOR.search(question)
        
        if temporal_match or anchor_match:
            axis = "document_time" if self.RE_DOC_TIME.search(question) else "event_time"
            slot = "DATE" if temporal_match else "VALUE"
            
            relation = "MATCH"
            if self.RE_EARLIEST.search(question):
                relation = "EARLIEST"
            elif self.RE_LATEST.search(question):
                relation = "LATEST"
                
            anchor = ""
            precision = ""
            if anchor_match:
                # Basic normalization for tests, e.g., "April 3" -> "2024-04-03" or just retain for lexical match
                # For P1A, we'll keep the raw string and do a loose string match in execution,
                # or normalize it if possible. Let's keep it simple for now.
                anchor = anchor_match.group(1).lower()
                precision = "DAY" if re.search(r'\d{1,2}', anchor) else "MONTH"
                
            concept = self._extract_concept(question)
            if not concept:
                return RouteDecision(route="HARD")
                
            return RouteDecision(
                route="TEMPORAL",
                subject_id=subject_id,
                target_concept=concept,
                answer_slot=slot,
                temporal_axis=axis,
                temporal_relation=relation,
                temporal_anchor=anchor,
                temporal_precision=precision
            )
            
        # 3. STATE_LATEST
        if self.RE_STATE_LATEST.search(question):
            concept = self._extract_concept(question)
            if not concept:
                return RouteDecision(route="HARD")
            return RouteDecision(
                route="STATE_LATEST",
                subject_id=subject_id,
                target_concept=concept,
                answer_slot="VALUE"
            )
            
        # 4. EXACT
        # If it's a simple "What is <concept>?" or "What <concept>..."
        if question.lower().startswith("what"):
            concept = self._extract_concept(question)
            if concept:
                return RouteDecision(
                    route="EXACT",
                    subject_id=subject_id,
                    target_concept=concept,
                    answer_slot="VALUE"
                )
                
        return RouteDecision(route="HARD")

    def _resolve_subject(self, question: str) -> str:
        # Check explicit third party
        tp_matches = self.RE_THIRD_PARTY.finditer(question)
        for tp_match in tp_matches:
            noun = tp_match.group(1).lower()
            subject_id = f"third_party:{noun}"
            if self.subject_postings and subject_id in self.subject_postings:
                return subject_id
            
        if self.RE_PRIMARY.search(question):
            return "primary_user"
            
        return "primary_user"

    def _extract_concept(self, question: str) -> str:
        """
        Extremely naive concept extractor for P1A.
        Removes stopwords and question words.
        """
        q = question.lower()
        q = re.sub(r'[^a-z0-9\s]', '', q)
        
        # Remove common question wrappers
        stopwords = {
            "what", "when", "where", "who", "why", "how", "is", "are", "was", "were", 
            "do", "does", "did", "the", "a", "an", "my", "patient", "user", "i", "me",
            "current", "latest", "now", "documented", "recorded", "reported", "mentioned",
            "started", "start", "began", "first", "appeared", "onset", "occurred", "earliest",
            "date", "time", "status", "value", "level", "taking", "took", "visit", "on", "in", "at",
            "of", "for", "to", "with", "recent", "about", "there", "any"
        }
        
        words = q.split()
        concept_words = [w for w in words if w not in stopwords and not w.isdigit()]
        
        # Also remove months
        months = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
                  "january", "february", "march", "april", "june", "july", "august", "september",
                  "october", "november", "december"}
        concept_words = [w for w in concept_words if w not in months]
        
        if not concept_words:
            return ""
        return " ".join(concept_words)

