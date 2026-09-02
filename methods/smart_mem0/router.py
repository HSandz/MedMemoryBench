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
    temporal_year: Optional[int] = None
    temporal_month: Optional[int] = None
    temporal_day: Optional[int] = None

class DeterministicRouter:
    """
    100% deterministic (zero-LLM) query router for SmartMem0 P1A.
    Fails open to HARD if there is any ambiguity.
    """
    
    # Generic subject regexes
    RE_PRIMARY = re.compile(r'\b(i|me|my|mine|we|our)\b', re.IGNORECASE)
    RE_THIRD_PARTY = re.compile(r'\bmy\s+([a-z]+)\b', re.IGNORECASE)
    
    # Structural FAST checks - we only allow exact matches for fast routes
    RE_VISIBLE_OPTIONS = re.compile(r'(?m)^\s*[A-H][.)]\s+')
    
    RE_HARD = re.compile(
        r'\b(compare|compared|why|how|should|options|or|cause|because|result in|result from|need to|do i need|can i|could\s+.*?be related|does this mean|recommend|better|best|versus|difference|effect|related to)\b|\ba\s*,\s*b\s*,\s*c\b',
        re.IGNORECASE
    )
    
    RE_TEMPORAL_DATE = re.compile(r'\b(when|what date|what time)\b', re.IGNORECASE)
    
    # Note: Added 'start' based on previous patch, kept inside the group
    RE_DOC_TIME = re.compile(r'\b(documented|recorded|reported|mentioned|visit)\b', re.IGNORECASE)
    RE_EVENT_TIME = re.compile(r'\b(started|start|began|first appeared|onset|occurred|taking|took)\b', re.IGNORECASE)
    
    RE_EARLIEST = re.compile(r'\b(first|earliest|initial|started|start)\b', re.IGNORECASE)
    RE_LATEST_TEMP = re.compile(r'\b(latest|last|most recent)\b', re.IGNORECASE)
    
    RE_STATE_LATEST = re.compile(r'\b(current|latest|now)\b', re.IGNORECASE)
    
    MONTHS = {
        'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
        'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
        'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
        'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
    }
    
    def __init__(self, subject_postings: dict = None):
        self.subject_postings = subject_postings or {}

    def _parse_temporal_anchor(self, question: str):
        # Look for combinations of Month Day, Year or YYYY-MM-DD
        # Examples: "April 3", "February 2024", "2024-03-20", "2024"
        
        # 1. YYYY-MM-DD
        m = re.search(r'\b(\d{4})-(\d{2})-(\d{2})\b', question)
        if m:
            return m.group(0), int(m.group(1)), int(m.group(2)), int(m.group(3)), "DAY"
            
        # 2. YYYY-MM
        m = re.search(r'\b(\d{4})-(\d{2})\b', question)
        if m:
            return m.group(0), int(m.group(1)), int(m.group(2)), None, "MONTH"
            
        # 3. Month Day, Year
        month_pattern = r'\b(?:on\s+|in\s+|during\s+|at\s+)?(?:the\s+)?(' + '|'.join(self.MONTHS.keys()) + r')\b\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b'
        m = re.search(month_pattern, question, re.IGNORECASE)
        if m:
            month_str, day_str, year_str = m.group(1).lower(), m.group(2), m.group(3)
            return m.group(0), int(year_str) if year_str else None, self.MONTHS[month_str], int(day_str), "DAY"
            
        # 4. Month Year
        month_yr_pattern = r'\b(?:on\s+|in\s+|during\s+|at\s+)?(?:the\s+)?(' + '|'.join(self.MONTHS.keys()) + r')\b(?:,?\s+(\d{4}))\b'
        m = re.search(month_yr_pattern, question, re.IGNORECASE)
        if m:
            month_str, year_str = m.group(1).lower(), m.group(2)
            return m.group(0), int(year_str), self.MONTHS[month_str], None, "MONTH"
            
        # 5. Year only (very basic, requires context like "in 2024")
        m = re.search(r'\b(?:in|during)\s+(\d{4})\b', question, re.IGNORECASE)
        if m:
            return m.group(0), int(m.group(1)), None, None, "YEAR"
            
        return None, None, None, None, ""

    def route_query(self, question: str) -> RouteDecision:
        # STRICT WHITELIST APPROACH
        # First filter out clear HARD triggers
        if self.RE_HARD.search(question) or self.RE_VISIBLE_OPTIONS.search(question):
            return RouteDecision(route="HARD")
            
        subject_id = self._resolve_subject(question)
        if subject_id == "HARD":
            return RouteDecision(route="HARD")
            
        anchor_str, year, month, day, precision = self._parse_temporal_anchor(question)
        temporal_date = self.RE_TEMPORAL_DATE.search(question)
        
        # 2. TEMPORAL
        if temporal_date or anchor_str or self.RE_EARLIEST.search(question) or self.RE_LATEST_TEMP.search(question):
            axis = "document_time" if self.RE_DOC_TIME.search(question) else "event_time"
            slot = "DATE" if temporal_date else "VALUE"
            
            relation = "MATCH"
            if self.RE_EARLIEST.search(question):
                relation = "EARLIEST"
            elif self.RE_LATEST_TEMP.search(question):
                relation = "LATEST"
                
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
                temporal_anchor=anchor_str or "",
                temporal_year=year,
                temporal_month=month,
                temporal_day=day,
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
        """Resolve the subject of the question deterministically."""
        tp_matches = self.RE_THIRD_PARTY.finditer(question)
        for tp_match in tp_matches:
            noun = tp_match.group(1).lower()
            subject_id = f"third_party:{noun}"
            if self.subject_postings and subject_id in self.subject_postings:
                return subject_id
                
        if self.RE_PRIMARY.search(question):
            return "primary_user"
            
        # If no explicit owner, check if there's exactly 1 eligible subject in ledger
        if self.subject_postings:
            if len(self.subject_postings) == 1:
                return list(self.subject_postings.keys())[0]
            
        return "HARD"

    def _extract_concept(self, question: str) -> str:
        q = question.lower()
        q = re.sub(r'[^a-z0-9\s]', '', q)
        
        stopwords = {
            "what", "when", "where", "who", "why", "how", "is", "are", "was", "were", 
            "do", "does", "did", "the", "a", "an", "my", "patient", "user", "i", "me",
            "current", "latest", "now", "documented", "recorded", "reported", "mentioned",
            "started", "start", "began", "first", "appeared", "onset", "occurred", "earliest",
            "date", "time", "status", "value", "level", "taking", "took", "visit", "on", "in", "at",
            "of", "for", "to", "with", "recent", "about", "there", "any", "mine", "we", "our"
        }
        
        words = q.split()
        concept_words = [w for w in words if w not in stopwords and not w.isdigit()]
        
        months = set(self.MONTHS.keys())
        concept_words = [w for w in concept_words if w not in months]
        
        if not concept_words:
            return ""
        return " ".join(concept_words)

