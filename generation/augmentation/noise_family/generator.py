"""Family/friends noise session generator.

Generates family/friend roles for user personas and produces continuous
health consultation dialogues for each role. Supports inter-session
continuity via key-point summaries for context passing.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import FamilyNoiseConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Prompts
# ============================================================================

ROLE_GENERATION_PROMPT = '[Important Statement] This task is part of the medical artificial intelligence academic research project and is used to build the medical conversation benchmark data set (MedMemoryBench). All generated content is **completely fictional character creations** and is intended only for:\n1. Training and evaluating the conversational capabilities of medical AI assistants\n2. Test the AI system’s understanding of complex health consultation scenarios\n3. Academic paper publication and scientific research purposes\n\nAll generated content will not be used for actual medical decision-making, please feel free to generate fictional character data that meets the requirements.\n\n---\n\nPlease generate the roles of relatives and friends around the user based on the user\'s portrait information. Each role must have a complete and detailed health file that can support long-term and in-depth health consultation.\n\n## User portrait\n{persona_info}\n\n## Generate requirements\n\n### 1. Basic information (each role)\nGenerate {num_roles} roles, each containing:\n- relationship: relationship with the user (father/mother/spouse/in-laws/friends/children, etc.)\n- name: colloquial title (such as "my dad", "Lao Zhang", "Xiao Li")\n- age_range: specific age range (such as "62-65 years old")\n- occupation: occupation or status\n- Personality: Personality characteristics (affecting attitude and compliance with medical treatment)\n- Lifestyle_habits: Overview of lifestyle habits (diet, exercise, work and rest, tobacco and alcohol, etc.)\n\n### 2. Health profile (3-5 health problems for each character, this is the key!)\nEach health_condition must contain the following details:\n\n**Basic information:**\n- condition_name: the exact medical name of the disease/symptom\n- icd_category: Disease categories (such as cardiovascular, endocrine, orthopedics, respiratory system, etc.)\n- severity: severity (mild/moderate/severe) + specific instructions\n- duration: exact duration of illness\n\n**Condition details:**\n- diagnosis_history: diagnosis process (when, how it was discovered, what examinations were performed)\n- symptoms_detail: specific symptom description (frequencyrate, extent, incentives, mitigating factors)\n- recent_changes: recent changes (symptom changes in the last 1-2 weeks)\n- lab_results: recent inspection/laboratory results (specific values are required)\n\n**Treatment status:**\n- Medications: Detailed medication plan, format: drug name + dosage + frequency + duration of medication + effect evaluation\n- treatment_history: past treatment history (what treatments have been done and the effects)\n- doctor_recommendations: recommendations given by doctors\n\n**Focus:**\n- concerns: the issues that family members are most worried about (concrete)\n- questions_to_ask: List of specific questions that the family wants to ask (3-5)\n- upcoming_events: upcoming medical events (review, surgery, dressing change, etc.)\n\n### 3. Role diversity requirements\n- Elderly people (parents/grandparents): Chronic disease management (hypertension, diabetes, coronary heart disease, osteoarthritis, chronic obstructive pulmonary disease, etc.), often coexisting with multiple diseases\n- Middle-aged people (spouse/siblings): sub-health, occupational diseases, stress-related diseases (cervical spondylosis, fatty liver, stomach problems, anxiety and depression)\n- Children/Adolescents (Kids): Growth and Development, Immunity, Allergy, Myopia, Mental Health\n- Friends and colleagues: It can be special diseases (tumor rehabilitation, autoimmune diseases, rare diseases, etc.)\n\n### 4. Health problem correlation\nHealth issues in the same role should be intrinsically related, for example:\n- Diabetes → Diabetic retinopathy risk → Diabetic nephropathy monitoring\n- Hypertension → Coronary heart disease → Hyperlipidemia\n- Long-term sitting → Cervical spondylosis + Lumbar disc herniation + Fatty liver\n\n## Output format\nReturn a JSON array to ensure that each character\'s health file is detailed, authentic, and credible enough to support at least 20 rounds of in-depth medical consultation conversations.\n\n```json\n[\n  {{\n    "relationship": "father",\n    "name": "my dad",\n    "age_range": "62-65 years old",\n    "occupation": "retired worker",\n    "personality": "Stubborn, afraid of trouble, and doesn\'t like going to the hospital",\n    "lifestyle_habits": "Eats a salty diet, doesn\'t like to exercise, smokes for 30 years (1 pack a day), drinks occasionally",\n    "health_conditions": [\n      {{\n        "condition_name": "Type 2 diabetes",\n        "icd_category": "Endocrine metabolism",\n        "severity": "Moderate - unstable blood sugar control",\n        "duration": "5 years since diagnosis",\n        "diagnosis_history": "In 2019, a physical examination at the work unit found that fasting blood sugar was elevated (7.8mmol/L), and I later went to the hospital for a glucose tolerance test to confirm the diagnosis.",\n        "symptoms_detail": "Occasionally thirsty and excessive drinking, blurred vision, and occasional numbness in both feet",\n        "recent_changes": "Blood sugar has fluctuated greatly in the past week, and blood sugar after meals often exceeds 12mmol/L",\n        "lab_results": "The latest (2 weeks ago): fasting blood glucose 8.2mmol/L, glycosylated hemoglobin 7.8%, urine microalbumin 30mg/L",\n        "medications": "Metformin extended-release tablets 0.5g bid (after breakfast and dinner), taken for 3 years, the effect is average; glimepiride 2mg qd (before breakfast), added for 1 month",\n        "treatment_history": "Initially, metformin alone was used, and the control was acceptable. Blood sugar began to rise last year, and the doctor recommended combined medication.",\n        "doctor_recommendations": "Control diet, monitor blood sugar, and review glycation in 3 months",\n        "concerns": "Concern about developing complications of diabetes, especially the eyes and kidneys",\n        "questions_to_ask": ["Is blood sugar fluctuations due to insufficient medicine?", "Ankle numbnessIs it a sign of complications?", "Should I take insulin?", "How to control diet specifically?"],\n        "upcoming_events": "Appointment for fundus examination next month"\n      }}\n    ]\n  }}\n]\n```\n\nOnly return the JSON array and no other content.'

FAMILY_USER_SYSTEM_PROMPT = 'You are a user consulting an online doctor about the health of a loved one. You know your loved one’s condition very well and can tell you detailed symptoms, medications, and examinations.\n\n## Your identity\n{user_identity}\n\n## Information about relatives you want to consult\n- Relationship: {family_relationship}\n- Title: {family_name}\n- Age: {family_age}\n- Occupation: {family_occupation}\n- Personality traits: {family_personality}\n\n## The relative’s complete health record\n{health_conditions_detail}\n\n## Focus of this consultation\n{current_consultation_focus}\n\n## Summary of past consultation records\n{past_consultations}\n\n## Dialogue requirements\n\n### 1. Ask questions about professionalism\n- Ability to accurately describe the location, nature, frequency, duration, and triggers of symptoms\n- Be able to tell the specific test result values (such as "fasting blood glucose 8.2", "blood pressure 150/95")\n- Be able to tell the complete medication plan (drug name, dosage, taking time)\n- Able to describe the effects and adverse reactions of medication\n\n### 2. Consult continuity\n- If this is the first consultation: give a complete introduction to the background of the condition and ask the questions you are most concerned about\n- If it is a follow-up consultation:\n  * First give feedback on the implementation and effect of the last doctor’s advice\n  * Describe the latest changes in the condition\n  * Ask further questions based on new information\n\n### 3. Conversational style\n- Speak like a family member who truly cares about their loved one, reflecting anxiety and concern\n- Can express confusion ("I don\'t quite understand...", "Is this indicator...")\n- You can ask for details ("What should be done specifically?", "Can this medicine be taken with XX")\n- Each reply should be 2-4 sentences, which can include multiple related questions\n- Use colloquial expressions appropriately, but the medical information must be accurate\n\n### 4. Focus on the key points of consultation\nQuestions must be closely related to the focus of this consultation "{current_consultation_focus}", but can be naturally related to other health issues (because chronic diseases often interact with each other)Influence).\n\nPlease reply directly as the user without adding any role tags or prefixes.'

FAMILY_DOCTOR_SYSTEM_PROMPT = """You are a senior online consultation doctor with rich clinical experience. You are responding to a family member's inquiry about a loved one's health.

## Basic patient information
- Relationship with the consultant: {family_relationship} of the consultant
- Title: {family_name}
- Age: {family_age}

## Patient complete health file
{health_conditions_detail}

## Focus of this consultation
{current_consultation_focus}

## Reply to request

### 1. Professionalism
- The reply should reflect the understanding of the patient's overall condition, not talk in general terms
- Can give targeted suggestions based on the patient's specific examination results and medication status
- Be easy to understand when explaining medical concepts, but be precise in terminology
- Explain the medical rationale behind the recommendation when necessary

### 2. The reply content can include
- **Explanation of condition**: Explain the meaning of symptoms/indicators and possible causes
- **Medication Guidance**: Specific medication adjustment suggestions (dose, time, precautions)
- **Lifestyle**: Specific suggestions on diet, exercise, and work and rest (must be actionable)
- **Monitoring recommendations**: Indicators that need attention, monitoring frequency, and recording methods
- **Review Arrangement**: When to review and what items to check
- **Warning Signs**: Dangerous symptoms to be alert to, situations that require immediate medical attention
- **Ask for details**: If there is insufficient information, you can ask for more details to give accurate suggestions.

### 3. Safety principles
- For serious cases, clearly recommend offline medical treatment, do not just give online advice
- Do not change the medication plan at will. For major adjustments, it is recommended to consult the attending doctor.
- Reminder of medication contraindications and interactions

### 4. Reply style
- Speak professionally but kindly, like a responsible doctor
- Reply in 3-6 sentences, with substantial content but not lengthy
- Use bulleted descriptions, but do not over-format them
- Key values and precautions should be clear

Please reply directly as a physician without adding any role tags or prefixes."""

CONSULTATION_FOCUS_PROMPT = 'Plan the specific focus topics for this consultation based on your loved one’s health records and past consultation history.\n\n## Relatives’ health records\n{health_conditions_detail}\n\n## Past consultation history\n{past_consultations}\n\n## Consult key planning strategies\n\n### If this is your first consultation:\nChoose one of the following as your entry point:\n1. The most pressing problem (symptoms that have become significantly worse or changed recently)\n2. The most troublesome problems in daily life\n3. Preparations for upcoming medical events (review, surgery, etc.)\n4. The risk of complications that family members are most worried about\n\n### If it is a follow-up consultation (with consultation history):\nSelect by priority:\n1. Implementation feedback since the last consultation + emerging situations\n2. In-depth questioning of questions that were not fully answered last time\n3. New changes and new symptoms of the condition\n4. Interpretation of review results or next treatment plan\n5. Follow up on previously mentioned matters of concern\n\n### Topic specific requirements:\nTopics must be specific and clear, including:\n- Specific symptoms/indicators/drug names\n- Time nodes or description of changes\n- Clear purpose of consultation\n\n**Good example:**\n- "My father\'s post-meal blood sugar has often exceeded 12 in the past week. I would like to ask if the medication plan needs to be adjusted."\n- "Last time, after increasing the dose of metformin recommended by the doctor, my stomach felt uncomfortable. I would like to ask if there are any alternatives."\n- "I will have a fundus examination next week, and I would like to know the precautions and possible results of the examination in advance."\n- "My mother\'s bone density has dropped again. Which brand of calcium tablets is better?"\n\n**Bad Example:**\n- "Blood sugar problems" (too broad)\n- "Diabetes consultation" (no specific point)\n- "Not feeling well" (unclear)\n\n## Output\nPlease return directly to a specific consultation topic (one sentence, 15-40 words) without any other content.'

EXTRACT_SUMMARY_PROMPT = 'Please extract a summary of key information from the conversation below for contextual reference in subsequent consultations.\n\n## Family information\n- Relationship: {family_relationship}\n- Title: {family_name}\n\n## Focus of this consultation\n{consultation_focus}\n\n##Conversation content\n{dialogue_text}\n\n## Requirements\nExtract 3-5 key points, including:\n1. Main issues of this consultation\n2. Specific advice and guidance given by doctors\n3. Matters requiring follow-up attention or review\n4. Suggestions on medication or lifestyle adjustments\n5. Next action plan (if any)\n\nPlease describe in concise Chinese, with one line for each point, starting with "-". Only the summary content is returned, no other explanation is required.'


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HealthCondition:
    """Health condition/disease with detailed health profile."""

    # Basic info
    condition_name: str  # Exact medical name of the disease/symptom
    icd_category: str = ""  # Disease category (cardiovascular, endocrine, etc.)
    severity: str = ""  # Severity level with details
    duration: str = ""  # Exact duration of illness

    # Condition details
    diagnosis_history: str = ""  # Diagnosis history
    symptoms_detail: str = ""  # Detailed symptom description
    recent_changes: str = ""  # Recent changes
    lab_results: str = ""  # Recent lab/test results

    # Treatment info
    medications: str = ""  # Detailed medication plan
    treatment_history: str = ""  # Past treatment history
    doctor_recommendations: str = ""  # Previous doctor recommendations

    # Concerns
    concerns: str = ""  # Family member's primary concerns
    questions_to_ask: List[str] = field(default_factory=list)  # Specific questions to consult
    upcoming_events: str = ""  # Upcoming medical events

    # Legacy field
    current_status: str = ""  # Current status (backward compatible)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition_name": self.condition_name,
            "icd_category": self.icd_category,
            "severity": self.severity,
            "duration": self.duration,
            "diagnosis_history": self.diagnosis_history,
            "symptoms_detail": self.symptoms_detail,
            "recent_changes": self.recent_changes,
            "lab_results": self.lab_results,
            "medications": self.medications,
            "treatment_history": self.treatment_history,
            "doctor_recommendations": self.doctor_recommendations,
            "concerns": self.concerns,
            "questions_to_ask": self.questions_to_ask,
            "upcoming_events": self.upcoming_events,
            "current_status": self.current_status,
        }

    def to_summary(self) -> str:
        """Convert to brief summary text."""
        parts = [f"【{self.condition_name}】"]
        if self.severity:
            parts.append(f"{self.severity}")
        if self.duration:
            parts.append(f"，病程{self.duration}")
        if self.current_status:
            parts.append(f"，{self.current_status}")
        if self.medications:
            parts.append(f"。用药：{self.medications}")
        if self.concerns:
            parts.append(f"。家属担心：{self.concerns}")
        return "".join(parts)

    def to_detail(self) -> str:
        """Convert to detailed health profile text."""
        lines = [f"### {self.condition_name}"]

        # Basic info
        basic_info = []
        if self.icd_category:
            basic_info.append(f"分类：{self.icd_category}")
        if self.severity:
            basic_info.append(f"程度：{self.severity}")
        if self.duration:
            basic_info.append(f"病程：{self.duration}")
        if basic_info:
            lines.append(" | ".join(basic_info))

        if self.diagnosis_history:
            lines.append(f"**诊断经过**：{self.diagnosis_history}")

        if self.symptoms_detail:
            lines.append(f"**症状表现**：{self.symptoms_detail}")

        if self.recent_changes:
            lines.append(f"**近期变化**：{self.recent_changes}")

        if self.lab_results:
            lines.append(f"**检查Result**：{self.lab_results}")

        if self.medications:
            lines.append(f"**用药方案**：{self.medications}")

        if self.treatment_history:
            lines.append(f"**治疗史**：{self.treatment_history}")

        if self.doctor_recommendations:
            lines.append(f"**医生建议**：{self.doctor_recommendations}")

        if self.concerns:
            lines.append(f"**家属担心**：{self.concerns}")

        if self.questions_to_ask:
            lines.append(f"**想咨询的Question**：" + "；".join(self.questions_to_ask))

        if self.upcoming_events:
            lines.append(f"**近期安排**：{self.upcoming_events}")

        return "\n".join(lines)


@dataclass
class FamilyRole:
    """Family/friend role."""

    role_id: int  # Role ID (unique within the persona)
    persona_id: int  # Parent user persona ID
    relationship: str  # Relationship to the user
    name: str  # Informal name/title
    age_range: str  # Age range
    occupation: str  # Occupation
    personality: str  # Personality traits
    health_conditions: List[HealthCondition]  # List of health conditions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role_id": self.role_id,
            "persona_id": self.persona_id,
            "relationship": self.relationship,
            "name": self.name,
            "age_range": self.age_range,
            "occupation": self.occupation,
            "personality": self.personality,
            "health_conditions": [hc.to_dict() for hc in self.health_conditions],
        }

    def get_health_conditions_detail(self) -> str:
        """Get detailed health profile text."""
        if not self.health_conditions:
            return "No detailed health records yet"

        lines = []
        for i, hc in enumerate(self.health_conditions, 1):
            lines.append(f"## 健康Question {i}")
            lines.append(hc.to_detail())
            lines.append("")  # blank line separator

        return "\n".join(lines)


@dataclass
class FamilyNoiseMessage:
    """Family consultation session message."""

    turn: int
    role: str  # "user" or "assistant"
    content: str
    agent_type: str  # "user_agent" or "doctor_agent"


@dataclass
class FamilyNoiseSession:
    """Family/friends noise session."""

    noise_family_id: int  # Noise identifier
    noise_type: str  # "family_health_consultation"
    persona_id: int  # Parent user persona ID
    family_role: FamilyRole  # Family role info
    health_issue: str  # Health issue for this consultation
    turn_count: int
    messages: List[FamilyNoiseMessage]
    knowledge_points: List[Dict[str, Any]]
    session_summary: str  # Session summary (context for subsequent sessions)
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Process messages: may be FamilyNoiseMessage objects or dicts
        messages_list = []
        for msg in self.messages:
            if isinstance(msg, FamilyNoiseMessage):
                messages_list.append({
                    "turn": msg.turn,
                    "role": msg.role,
                    "content": msg.content,
                    "agent_type": msg.agent_type,
                })
            else:
                messages_list.append(msg)
        
        return {
            "noise_family_id": self.noise_family_id,
            "noise_type": self.noise_type,
            "persona_id": self.persona_id,
            "family_role": self.family_role.to_dict(),
            "health_issue": self.health_issue,
            "turn_count": self.turn_count,
            "messages": messages_list,
            "knowledge_points": self.knowledge_points,
            "session_summary": self.session_summary,
            "created_at": self.created_at,
        }


# ============================================================================
# Generator
# ============================================================================

class FamilyDialogueGenerator:
    """Family/friends noise session generator.

    Generates family roles for user personas and produces continuous
    health consultation dialogues for each role.
    """

    def __init__(self, config: Optional[FamilyNoiseConfig] = None):
        """Initialize generator.

        Args:
            config: Configuration object.
        """
        self.config = config or FamilyNoiseConfig()
        self._load_api_config()
        self._client = None

        # Role and session records
        self._roles_by_persona: Dict[int, List[FamilyRole]] = {}
        self._role_past_summaries: Dict[str, List[str]] = {}  # role_key -> summaries

        # Per-role knowledge points (role_key -> knowledge_points)
        self._role_knowledge_points: Dict[str, List[Dict[str, Any]]] = {}

        logger.info("[FamilyDialogueGenerator] Initialization complete")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Sessions per role: {self.config.sessions_per_role}")

    def _load_api_config(self) -> None:
        """Load API config."""
        import os
        from dotenv import load_dotenv

        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)  # Force override existing env vars

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        model = self.config.model or os.getenv("LLM_MODEL") or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        self.model = model.replace("openai/", "") if model.startswith("openai/") else model

    @property
    def client(self):
        """Get OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def _get_completion_kwargs(self, max_tokens: int) -> Dict[str, Any]:
        """Get completion parameters compatible with different models."""
        legacy_model_prefixes = (
            "gpt-3.5",
            "gpt-4-",
            "text-davinci",
            "text-curie",
            "text-babbage",
            "text-ada",
        )

        if any(self.model.startswith(prefix) for prefix in legacy_model_prefixes):
            return {"max_tokens": max_tokens}
        else:
            return {"max_completion_tokens": max_tokens}

    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        caller: str = "unknown",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Unified LLM call method with token tracking.

        Args:
            messages: LLM message list.
            caller: Caller identifier for tracking.
            temperature: Temperature parameter (defaults to config value).
            max_tokens: Max token count (defaults to config value).

        Returns:
            LLM response content.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from app.services.token_tracker import get_token_tracker

        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            **self._get_completion_kwargs(tokens),
        )

        # Track token usage
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            tracker = get_token_tracker()
            tracker.set_model(self.model)
            tracker.track(
                prompt_tokens=usage.prompt_tokens or 0,
                completion_tokens=usage.completion_tokens or 0,
                total_tokens=usage.total_tokens or 0,
                caller=f"noise_family.{caller}",
            )

        content = response.choices[0].message.content
        return content.strip() if content else ""

    def load_personas(self) -> List[Dict[str, Any]]:
        """Load user persona data.

        Returns:
            List of user personas.
        """
        personas_path = Path(self.config.data_dir) / self.config.personas_filename
        logger.info(f"[FamilyDialogueGenerator] LoadUser persona: {personas_path}")

        with open(personas_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        personas = data.get("personas", [])
        logger.info(f"  Loaded {len(personas)} user personas")
        return personas

    async def generate_family_roles(self, persona: Dict[str, Any]) -> List[FamilyRole]:
        """Generate family/friend roles for a single user persona.

        Args:
            persona: User persona data.

        Returns:
            List of generated family roles with detailed health conditions.
        """
        persona_id = persona["persona_id"]
        logger.info(f"[FamilyDialogueGenerator] Generating family roles for persona {persona_id}")

        # Build user persona info
        base_info = persona.get("base_info", {})
        enriched = persona.get("enriched_data", {})

        persona_info = f"""
- 疾病类型: {base_info.get('type_name', 'N/A')}
- 性别: {base_info.get('gender', 'N/A')}
- 年龄范围: {enriched.get('age_range', 'N/A')}
- 职业: {enriched.get('occupation_detail', 'N/A')}
- 背景故事: {enriched.get('background_story', 'N/A')[:300]}...
"""

        prompt = ROLE_GENERATION_PROMPT.format(
            persona_info=persona_info,
            num_roles=self.config.num_family_roles,
        )

        try:
            content = self._call_llm(
                [{"role": "user", "content": prompt}],
                caller="generate_family_roles",
                max_tokens=8000,
            )

            # Check if response content is empty
            if not content:
                logger.warning("[FamilyDialogueGenerator] LLM returned empty content, using default roles")
                default_roles = self._generate_default_roles(persona_id)
                self._roles_by_persona[persona_id] = default_roles
                return default_roles

            logger.debug(f"[FamilyDialogueGenerator] LLM raw response (first 500 chars): {content[:500]}")

            # Parse JSON - try multiple extraction methods
            json_content = content
            if "```json" in content:
                parts = content.split("```json")
                if len(parts) > 1:
                    json_content = parts[1].split("```")[0].strip()
            elif "```" in content:
                parts = content.split("```")
                if len(parts) > 1:
                    json_content = parts[1].split("```")[0].strip()

            # If extracted content is empty, try parsing original content directly
            if not json_content:
                logger.warning("[FamilyDialogueGenerator] JSON extraction result is empty, trying to parse original content directly")
                json_content = content

            # Try to find the start and end of JSON array
            if json_content and not json_content.startswith("["):
                start_idx = json_content.find("[")
                if start_idx != -1:
                    end_idx = json_content.rfind("]")
                    if end_idx != -1:
                        json_content = json_content[start_idx:end_idx + 1]

            logger.debug(f"[FamilyDialogueGenerator] Extracted JSON (first 500 chars): {json_content[:500] if json_content else 'empty'}")

            roles_data = json.loads(json_content)

            # Convert to FamilyRole objects
            roles = []
            for idx, role_data in enumerate(roles_data):
                # Parse health conditions list (supports detailed fields)
                health_conditions = []
                for hc_data in role_data.get("health_conditions", []):
                    # Process questions_to_ask field
                    questions = hc_data.get("questions_to_ask", [])
                    if isinstance(questions, str):
                        questions = [q.strip() for q in questions.split("；") if q.strip()]

                    hc = HealthCondition(
                        condition_name=hc_data.get("condition_name", ""),
                        icd_category=hc_data.get("icd_category", ""),
                        severity=hc_data.get("severity", "Moderate"),
                        duration=hc_data.get("duration", ""),
                        diagnosis_history=hc_data.get("diagnosis_history", ""),
                        symptoms_detail=hc_data.get("symptoms_detail", ""),
                        recent_changes=hc_data.get("recent_changes", ""),
                        lab_results=hc_data.get("lab_results", ""),
                        medications=hc_data.get("medications", "none"),
                        treatment_history=hc_data.get("treatment_history", ""),
                        doctor_recommendations=hc_data.get("doctor_recommendations", ""),
                        concerns=hc_data.get("concerns", ""),
                        questions_to_ask=questions,
                        upcoming_events=hc_data.get("upcoming_events", ""),
                        current_status=hc_data.get("current_status", ""),
                    )
                    health_conditions.append(hc)

                role = FamilyRole(
                    role_id=idx + 1,
                    persona_id=persona_id,
                    relationship=role_data.get("relationship", "family"),
                    name=role_data.get("name", "family"),
                    age_range=role_data.get("age_range", "unknown"),
                    occupation=role_data.get("occupation", "unknown"),
                    personality=role_data.get("personality", ""),
                    health_conditions=health_conditions,
                )
                roles.append(role)

                logger.info(f"  Role {idx+1}: {role.name}({role.relationship}), "
                           f"{len(health_conditions)} health questions")

            self._roles_by_persona[persona_id] = roles
            return roles

        except json.JSONDecodeError as e:
            logger.error(f"[FamilyDialogueGenerator] JSON ParseFailed: {e}")
            logger.error(f"[FamilyDialogueGenerator] Content to parse (first 1000 chars): {json_content[:1000] if 'json_content' in dir() and json_content else 'empty'}")
            logger.error(f"[FamilyDialogueGenerator] Raw response (first 1000 chars): {content[:1000] if 'content' in dir() and content else 'empty'}")
            # Return default roles
            default_roles = self._generate_default_roles(persona_id)
            self._roles_by_persona[persona_id] = default_roles
            return default_roles
        except Exception as e:
            logger.error(f"[FamilyDialogueGenerator] Role generation failed: {e}")
            # Return default roles
            default_roles = self._generate_default_roles(persona_id)
            self._roles_by_persona[persona_id] = default_roles
            return default_roles

    def _generate_default_roles(self, persona_id: int) -> List[FamilyRole]:
        """Generate default family roles with detailed health conditions."""
        default_configs = [
            {
                "relationship": "Father",
                "name": "my dad",
                "age_range": "62-65 years old",
                "occupation": "retired worker",
                "personality": "Stubborn, afraid of trouble, and doesn’t like going to the hospital",
                "health_conditions": [
                    HealthCondition(
                        condition_name="type 2 diabetes",
                        icd_category="endocrine metabolism",
                        severity="Moderate - Unstable blood sugar control",
                        duration="5 years since diagnosis",
                        diagnosis_history="In 2019, a physical examination at work found that fasting blood sugar was elevated (7.8mmol/L), and I later went to the hospital for a glucose tolerance test to confirm the diagnosis.",
                        symptoms_detail="Occasionally thirsty and excessive drinking, blurred vision, and occasional numbness in both feet",
                        recent_changes="Blood sugar has fluctuated greatly in the past week, and blood sugar after meals often exceeds 12mmol/L.",
                        lab_results="The most recent time (2 weeks ago): fasting blood glucose 8.2mmol/L, glycosylated hemoglobin 7.8%, urine microalbumin 30mg/L",
                        medications="Metformin extended-release tablets 0.5g bid (after breakfast and dinner), taken for 3 years; glimepiride 2mg qd (before breakfast), added for 1 month",
                        treatment_history="Initially, metformin alone was able to control the disease, but last year, when blood sugar started to rise, it was combined with metformin.",
                        doctor_recommendations="Control diet, monitor blood sugar, and review glycated hemoglobin in 3 months",
                        concerns="Worry about developing complications of diabetes, especially the eyes and kidneys",
                        questions_to_ask=["Is it because the blood sugar fluctuates greatly that the medicine is not enough?", "Is numbness in the feet a sign of complications?", "Do you need insulin?"],
                        upcoming_events="Make an appointment for a fundus examination next month",
                    ),
                    HealthCondition(
                        condition_name="hypertension",
                        icd_category="cardiovascular",
                        severity="Moderate - stage 2 hypertension",
                        duration="Diagnosed 8 years ago",
                        diagnosis_history="In 2016, I went to the hospital for dizziness and found that my blood pressure was 160/100mmHg, and I was diagnosed with hypertension.",
                        symptoms_detail="No obvious symptoms at ordinary times, but occasionally dizziness when tired or emotional",
                        recent_changes="The weather has changed recently and my blood pressure has fluctuated, occasionally exceeding 150/95 in the morning.",
                        lab_results="Review last month: blood pressure 142/88mmHg, electrocardiogram generally normal, renal function normal",
                        medications="Amlodipine besylate 5 mg qd (morning), taken for 5 years, control is acceptable",
                        treatment_history="Initially, the effect of irbesartan was not good, but later it was switched to amlodipine and the control was better.",
                        doctor_recommendations="Eat a low-salt diet, monitor blood pressure daily, and maintain emotional stability",
                        concerns="Worry about the side effects of long-term medication and the development of heart disease or cerebrovascular disease",
                        questions_to_ask=["Can I take blood pressure medicine for a long time?", "Are there any medicines with few side effects?", "How much blood pressure is considered to be well controlled?"],
                        upcoming_events="",
                    ),
                    HealthCondition(
                        condition_name="Lumbar disc herniation",
                        icd_category="orthopedics",
                        severity="Moderate - L4/L5 prominence",
                        duration="Diagnosed 5 years ago, recurring attacks",
                        diagnosis_history="In 2019, I went to the hospital with low back pain and radiating pain in my right leg. CT showed L4/L5 intervertebral disc herniation.",
                        symptoms_detail="Waist pain, aggravated by sitting for a long time, and a pulling sensation in the right leg when bending over",
                        recent_changes="Recently, I started to feel pain again after helping to move things, and the numbness in my right leg was obvious.",
                        lab_results="MRI last year: L4/L5 disc herniation, dural sac compression",
                        medications="Use diclofenac sodium analgesic patch during an attack, and take celecoxib orally when the pain is severe.",
                        treatment_history="I had 3 courses of physical therapy with average results. The doctor recommended surgery, but I didn’t want to do it",
                        doctor_recommendations="Avoid sitting and bending for long periods of time, strengthen your back muscles, and consider surgery in severe cases",
                        concerns="Afraid of the risks of surgery, but also worried about paralysis if the condition worsens",
                        questions_to_ask=["Is it okay without surgery?", "What are the conservative treatments?", "How to exercise daily"],
                        upcoming_events="",
                    ),
                ],
            },
            {
                "relationship": "Mother",
                "name": "my mother",
                "age_range": "60-63 years old",
                "occupation": "retired teacher",
                "personality": "Loves to worry, is easily anxious, and is sensitive to health questions",
                "health_conditions": [
                    HealthCondition(
                        condition_name="osteoporosis",
                        icd_category="Orthopedics/Endocrinology",
                        severity="Moderate - T-score -2.8",
                        duration="Diagnosed 4 years ago",
                        diagnosis_history="The bone density test during the physical examination in 2020 found that the T value of the lumbar spine at that time was -2.5",
                        symptoms_detail="I occasionally have dull pain in my lower back, and my height is 2cm shorter than when I was younger.",
                        recent_changes="The bone density dropped again during the last review, and the T-score dropped from -2.5 to -2.8.",
                        lab_results="Bone density 3 months ago: lumbar spine T-score -2.8, femoral neck T-score -2.3. Blood calcium is normal, vitamin D is low (18ng/ml)",
                        medications="Calcium D 600mg one tablet per day, alfacalcidol 0.25μg once per day",
                        treatment_history="I have been taking calcium supplements, but the effect is not obvious. The doctor recommended adding bisphosphonates.",
                        doctor_recommendations="Continue to supplement calcium and D, consider anti-osteoporosis drugs, and pay attention to preventing falls",
                        concerns="Very worried about fractures, especially about being bedridden after hip fracture",
                        questions_to_ask=["Can bone density be restored?", "What are the side effects of bisphosphonates?", "How to prevent fractures every day"],
                        upcoming_events="Make an appointment with the endocrinology clinic next week",
                    ),
                    HealthCondition(
                        condition_name="insomnia",
                        icd_category="neurological/psychological",
                        severity="mild to moderate",
                        duration="More than 2 years",
                        diagnosis_history="After retiring, Start developed sleep problems and had difficulty falling asleep. He often woke up at 3 a.m. and could not fall asleep.",
                        symptoms_detail="It takes 1-2 hours to fall asleep, easy to wake up in the middle of the night, difficult to fall back asleep after waking up early, and tired during the day",
                        recent_changes="Recently, I have been sleeping even worse because I am worried about osteoporosis.",
                        lab_results="No special inspection was done",
                        medications="Alprazolam 0.4 mg before bed (only when you really can't sleep), 2-3 times a week",
                        treatment_history="I have tried melatonin, jujube kernel pills, etc., but the effect is not obvious.",
                        doctor_recommendations="Keep a regular schedule, don’t look at your phone before going to bed, and take medication to help you sleep if necessary.",
                        concerns="Worried about being addicted to sleeping pills, but can’t sleep without taking them",
                        questions_to_ask=["How can I sleep well without taking medicine?", "Can sleeping pills be taken for a long time?", "Is there any way to treat it with Chinese medicine?"],
                        upcoming_events="",
                    ),
                    HealthCondition(
                        condition_name="Thyroid nodules",
                        icd_category="endocrine",
                        severity="Mild - TI-RADS Category 3",
                        duration="Found 1 year",
                        diagnosis_history="Last year's physical examination and B-ultrasound revealed a 0.8cm nodule in the left lobe of the thyroid gland, with a clear border and no calcification.",
                        symptoms_detail="No obvious symptoms, no pain or itching, no abnormalities in swallowing",
                        recent_changes="There was no obvious change in the nodule during reexamination half a year ago, it was still 0.8cm.",
                        lab_results="Thyroid function is normal: TSH 2.1mIU/L, FT3 and FT4 are both normal. B-ultrasound: left lobe nodule 0.8×0.6cm, TI-RADS category 3",
                        medications="none",
                        treatment_history="Doctors recommend regular follow-up observations",
                        doctor_recommendations="Review B-ultrasound every 6 months to observe changes in nodules",
                        concerns="I am very worried that it will become cancerous. I am scared when I read about thyroid cancer on the Internet.",
                        questions_to_ask=["Can nodules become cancerous?", "Do you need a puncture?", "What should you pay attention to in your daily diet?"],
                        upcoming_events="Review B-ultrasound after 2 months",
                    ),
                ],
            },
            {
                "relationship": "spouse",
                "name": "husband",
                "age_range": "32-35 years old",
                "occupation": "programmer",
                "personality": "Workaholic, often works overtime, doesn’t pay much attention to health, and avoids medical treatment for illnesses.",
                "health_conditions": [
                    HealthCondition(
                        condition_name="cervical spondylosis",
                        icd_category="orthopedics",
                        severity="Mild - neck type",
                        duration="More than 2 years",
                        diagnosis_history="I have been working in front of the computer for a long time and my neck is sore. The X-ray taken last year showed that the physiological curvature of the cervical spine has straightened.",
                        symptoms_detail="Stiffness and soreness in the neck, aggravated by working with the head down for more than 2 hours, sometimes radiating to the shoulder",
                        recent_changes="I have been busy with projects and worked overtime recently, and my cervical spine pain has worsened, and I sometimes feel dizzy.",
                        lab_results="Cervical spine X-ray: The physiological curvature becomes straightened, and the C5-C6 intervertebral space is slightly narrowed. No MRI",
                        medications="Apply plaster when in pain and massage occasionally",
                        treatment_history="I bought a cervical massager and put it aside after using it a few times.",
                        doctor_recommendations="Correct your sitting posture and move your neck every hour. It is recommended to do cervical spine exercises.",
                        concerns="Worry about developing cervical disc herniation and affecting work",
                        questions_to_ask=["Will cervical spondylosis become more and more serious?", "What are some good treatments?", "Do you need an MRI?"],
                        upcoming_events="",
                    ),
                    HealthCondition(
                        condition_name="fatty liver",
                        icd_category="Gastroenterology",
                        severity="Mild",
                        duration="Found 1 year",
                        diagnosis_history="The B-ultrasound of the company's physical examination last year found that the transaminase was normal at that time",
                        symptoms_detail="No obvious symptoms, occasional right upper quadrant discomfort",
                        recent_changes="The recent physical examination showed that the transaminase was a bit high, ALT 68U/L.",
                        lab_results="B-ultrasound: The liver echo is enhanced, indicating mild fatty liver. ALT 68U/L (normal <40), AST 45U/L",
                        medications="none",
                        treatment_history="The doctor said I should lose weight and exercise, but I haven’t taken any action.",
                        doctor_recommendations="Control your weight, eat a low-fat diet, exercise more, and quit drinking",
                        concerns="Worry about developing cirrhosis or liver cancer",
                        questions_to_ask=["Should I take medicine if my transaminase is high?", "Can fatty liver be reversed?", "How often to review"],
                        upcoming_events="",
                    ),
                    HealthCondition(
                        condition_name="chronic gastritis",
                        icd_category="Gastroenterology",
                        severity="Mild - Superficial",
                        duration="3 years",
                        diagnosis_history="Irregular diet, frequent stomachache and bloating. Gastroscopy showed chronic superficial gastritis and negative Helicobacter pylori.",
                        symptoms_detail="Stomach bloating after eating, dull stomach pain during fasting, belching and acid reflux",
                        recent_changes="Recently, I have been working overtime and eating a lot of takeaways, and my stomach upsets have increased.",
                        lab_results="Gastroscopy 2 years ago: chronic superficial gastritis. HP(-).",
                        medications="Take omeprazole 20mg and aluminum magnesium carbonate when you have an upset stomach",
                        treatment_history="My stomach improved after taking medicine for a while, but it relapsed after stopping the medicine.",
                        doctor_recommendations="Eat regularly, eat less irritating foods, and avoid fasting for too long",
                        concerns="Worried about developing gastric ulcer or more serious Question",
                        questions_to_ask=["Do I need another gastroscopy?", "How can we cure it?", "Are there any side effects of taking omeprazole for a long time?"],
                        upcoming_events="",
                    ),
                ],
            },
            {
                "relationship": "friend",
                "name": "Lao Wang",
                "age_range": "38-42 years old",
                "occupation": "Sales Director",
                "personality": "Socializes a lot, likes to drink, lives an irregular life, and is careless about health",
                "health_conditions": [
                    HealthCondition(
                        condition_name="Gout/Hyperuricemia",
                        icd_category="endocrine metabolism",
                        severity="Moderate - had an acute attack",
                        duration="Diagnosed 2 years ago",
                        diagnosis_history="Two years ago, I experienced severe pain, redness and swelling in my right big toe at night. In the emergency room, my uric acid level was 580 μmol/L and I was diagnosed with gout.",
                        symptoms_detail="There are no obvious symptoms after the acute phase, but uric acid remains high.",
                        recent_changes="I've been doing a lot of socializing recently, and last week I had a dull pain in my toes, and I'm worried that it's going to happen again.",
                        lab_results="Last month: uric acid 495 μmol/L (normal <420), normal liver and kidney function",
                        medications="Febuxostat 40 mg once daily for more than 1 year",
                        treatment_history="Acute attacks are controlled with colchicine + indomethacin, followed by long-term use of uric acid-lowering drugs.",
                        doctor_recommendations="Strict food taboos, abstain from alcohol, drink plenty of water, and insist on taking medication",
                        concerns="I'm worried about it happening again. The pain is really unbearable.",
                        questions_to_ask=["Is it normal for uric acid to still be high after taking medicine?", "Can I have a drink once in a while?", "Do I need to take this medicine for the rest of my life?"],
                        upcoming_events="",
                    ),
                    HealthCondition(
                        condition_name="Hyperlipidemia",
                        icd_category="cardiovascular/metabolic",
                        severity="Mild - mainly elevated triglycerides",
                        duration="Found more than 1 year ago",
                        diagnosis_history="Physical examination found that it was related to long-term social drinking",
                        symptoms_detail="no obvious symptoms",
                        recent_changes="Triglycerides rose from 2.8 to 3.2mmol/L",
                        lab_results="Total cholesterol 5.8mmol/L, triglyceride 3.2mmol/L (normal <1.7), low-density lipoprotein 3.5mmol/L",
                        medications="No medication yet",
                        treatment_history="Doctors recommend lifestyle intervention for 3 months first",
                        doctor_recommendations="Stop drinking, eat a low-fat diet, and exercise more",
                        concerns="Worry about developing arteriosclerosis and coronary heart disease",
                        questions_to_ask=["Do you need to take lipid-lowering drugs?", "What are the symptoms of hyperlipidemia?", "Which is more dangerous, high triglycerides or high cholesterol?"],
                        upcoming_events="Check blood lipids again next month",
                    ),
                    HealthCondition(
                        condition_name="AnxietyStatus",
                        icd_category="psychological/neurological",
                        severity="Mild",
                        duration="About half a year",
                        diagnosis_history="High work pressure, heavy performance appraisal, nervousness, panic, insomnia, no formal medical treatment",
                        symptoms_detail="Easily nervous at work, flustered and shaking hands before meetings, sleep lightly at night, easily woken up",
                        recent_changes="I've been under a lot of performance pressure lately and my symptoms have gotten worse.",
                        lab_results="No inspection",
                        medications="No, I don’t want to take medicine",
                        treatment_history="I have bought some soothing health products, but the effect is not obvious.",
                        doctor_recommendations="Did not see a doctor",
                        concerns="I don’t want to be thought of as having psychological questions, and I’m also worried that taking medicine will affect my work.",
                        questions_to_ask=["Is this an anxiety disorder?", "Is it okay if I don’t take medicine?", "Is there any way to adjust it?"],
                        upcoming_events="",
                    ),
                ],
            },
            {
                "relationship": "child",
                "name": "Baby",
                "age_range": "4-5 years old",
                "occupation": "Kindergarten friend",
                "personality": "Lively and active, a bit picky about food, and resistant to injections and medicines",
                "health_conditions": [
                    HealthCondition(
                        condition_name="recurrent respiratory infections",
                        icd_category="Pediatrics/Respiratory",
                        severity="Mild - 6-8 colds per year",
                        duration="More than 1 year",
                        diagnosis_history="Start got sick frequently after entering kindergarten, especially during the change of seasons.",
                        symptoms_detail="Cold symptoms are mainly: runny nose, cough, fever, lasting 5-7 days each time",
                        recent_changes="I've had two colds this month, and my nose just started to run again.",
                        lab_results="Routine blood test from the last cold: normal white blood cells, high lymphocyte ratio",
                        medications="Symptomatic medicines for colds: children's acetaminophen, xanthanamine, Yitanjing, etc.",
                        treatment_history="I have been taking vitamins and probiotics for a while, but the effect is not obvious.",
                        doctor_recommendations="Strengthen nutrition, increase outdoor activities, and pay attention to protection during flu season",
                        concerns="I’m worried about whether my immunity is too low, which will affect my growth and development.",
                        questions_to_ask=["Do you need to check your immune function?", "How to improve immunity", "Should I get a flu shot?"],
                        upcoming_events="Preparing for an appointment with a child health department",
                    ),
                    HealthCondition(
                        condition_name="allergic rhinitis",
                        icd_category="Pediatrics/ENT",
                        severity="Mild - intermittent",
                        duration="more than half a year",
                        diagnosis_history="After getting up in the morning, I sneezed continuously and had a runny nose. The doctor said it was allergic rhinitis.",
                        symptoms_detail="Obvious in the morning and after exposure to dust, sneezing, runny nose, rubbing nose and eyes",
                        recent_changes="Symptoms have worsened during the recent change of seasons, and I feel a little stuffy when I sleep at night.",
                        lab_results="Allergens checked: dust mite level 2 positive, other negative",
                        medications="Rinse the nose with normal saline, and use mometasone furoate nasal spray if symptoms are severe",
                        treatment_history="Nasal washing has a certain effect, but the children are not very cooperative",
                        doctor_recommendations="Keep washing your nose, keep the room clean, and use nasal spray hormones when necessary",
                        concerns="I am worried about developing asthma. I heard that allergic rhinitis is related to asthma.",
                        questions_to_ask=["Will it develop into asthma?", "Do nasal spray hormones have any effect on children?", "Can it be cured?"],
                        upcoming_events="",
                    ),
                    HealthCondition(
                        condition_name="Picky eaters/underweight",
                        icd_category="Pediatrics/Nutrition",
                        severity="Mild - weight in the 15th percentile",
                        duration="More than 1 year",
                        diagnosis_history="I don’t like to eat vegetables and meat, and I eat less staple food. My body weight has always been underweight during physical examinations.",
                        symptoms_detail="Slowly eats, resists new foods, likes to eat snacks, and feels full after just a few bites of a meal",
                        recent_changes="My appetite has worsened recently after I got sick.",
                        lab_results="Height 105cm (P50), weight 15kg (P15). Hemoglobin 110g/L (slightly low), trace elements iron and zinc are low",
                        medications="Supplement vitamin AD, zinc gluconate oral solution",
                        treatment_history="Tried various methods, but the effect is not lasting",
                        doctor_recommendations="Develop good eating habits, increase food diversity, and supplement nutrients when necessary",
                        concerns="Worry about malnutrition affecting growth and intellectual development",
                        questions_to_ask=["Do you need any nutritional supplements?", "How to make children love to eat", "Will it affect your height?"],
                        upcoming_events="",
                    ),
                ],
            },
        ]

        roles = []
        for idx, cfg in enumerate(default_configs[:self.config.num_family_roles]):
            role = FamilyRole(
                role_id=idx + 1,
                persona_id=persona_id,
                relationship=cfg["relationship"],
                name=cfg["name"],
                age_range=cfg["age_range"],
                occupation=cfg["occupation"],
                personality=cfg["personality"],
                health_conditions=cfg["health_conditions"],
            )
            roles.append(role)

        return roles

    def _get_role_key(self, persona_id: int, role_id: int) -> str:
        """Get unique identifier key for a role."""
        return f"{persona_id}_{role_id}"

    def _get_past_consultations_for_role(self, persona_id: int, role_id: int) -> str:
        """Get past consultation records for a role."""
        role_key = self._get_role_key(persona_id, role_id)
        summaries = self._role_past_summaries.get(role_key, [])

        if not summaries:
            return "(This is the first time for this family to consult)"

        # Show recent consultation records
        recent = summaries[-5:]
        lines = [f"第{i+1}次咨询记录：" for i in range(len(recent))]
        result = []
        for i, summary in enumerate(recent):
            result.append(f"【第{i+1}次咨询】\n{summary}")

        return "\n\n".join(result)

    async def _select_consultation_focus(
        self,
        role: FamilyRole,
        session_idx: int,
    ) -> str:
        """Intelligently select the focus topic for this consultation.

        Args:
            role: Family role.
            session_idx: Session index (which consultation number).

        Returns:
            Focus topic for this consultation.
        """
        past_consultations = self._get_past_consultations_for_role(role.persona_id, role.role_id)
        health_conditions_detail = role.get_health_conditions_detail()

        prompt = CONSULTATION_FOCUS_PROMPT.format(
            health_conditions_detail=health_conditions_detail,
            past_consultations=past_consultations,
        )

        try:
            focus = self._call_llm(
                [{"role": "user", "content": prompt}],
                caller="_select_consultation_focus",
                max_tokens=500,
            )
            # Strip possible quotes
            focus = focus.strip('"\'')
            return focus
        except Exception as e:
            logger.warning(f"[FamilyDialogueGenerator] Consultation focus selection failed: {e}")
            # Round-robin selection of health conditions
            if role.health_conditions:
                idx = session_idx % len(role.health_conditions)
                hc = role.health_conditions[idx]
                return f"关于{role.name}的{hc.condition_name}的情况咨询"
            return f"关于{role.name}的健康状况咨询"

    async def _generate_user_turn(
        self,
        persona: Dict[str, Any],
        role: FamilyRole,
        consultation_focus: str,
        messages: List[FamilyNoiseMessage],
        turn: int,
    ) -> str:
        """Generate user turn."""
        enriched = persona.get("enriched_data", {})
        user_identity = f"年龄 {enriched.get('age_range', 'unknown')}，职业 {enriched.get('occupation_detail', 'unknown')}"

        past_consultations = self._get_past_consultations_for_role(role.persona_id, role.role_id)

        system_prompt = FAMILY_USER_SYSTEM_PROMPT.format(
            user_identity=user_identity,
            family_relationship=role.relationship,
            family_name=role.name,
            family_age=role.age_range,
            family_occupation=role.occupation,
            family_personality=role.personality,
            health_conditions_detail=role.get_health_conditions_detail(),
            current_consultation_focus=consultation_focus,
            past_consultations=past_consultations,
        )

        llm_messages = [{"role": "system", "content": system_prompt}]

        # Add dialogue history
        for msg in messages:
            llm_role = "assistant" if msg.agent_type == "user_agent" else "user"
            llm_messages.append({"role": llm_role, "content": msg.content})

        # First turn prompt
        if turn == 1:
            llm_messages.append({
                "role": "user",
                "content": f"请根据family的健康档案和本次咨询重点，向医生提出你的第一个Question。"
            })

        try:
            return self._call_llm(llm_messages, caller="_generate_user_turn")
        except Exception as e:
            logger.error(f"[FamilyDialogueGenerator] User turn generation failed: {e}")
            return f"医生您好，我想咨询一下{role.name}的{consultation_focus}。"

    async def _generate_doctor_turn(
        self,
        role: FamilyRole,
        consultation_focus: str,
        messages: List[FamilyNoiseMessage],
    ) -> str:
        """Generate doctor turn."""
        system_prompt = FAMILY_DOCTOR_SYSTEM_PROMPT.format(
            family_relationship=role.relationship,
            family_name=role.name,
            family_age=role.age_range,
            health_conditions_detail=role.get_health_conditions_detail(),
            current_consultation_focus=consultation_focus,
        )

        llm_messages = [{"role": "system", "content": system_prompt}]

        for msg in messages:
            llm_role = "user" if msg.agent_type == "user_agent" else "assistant"
            llm_messages.append({"role": llm_role, "content": msg.content})

        try:
            return self._call_llm(llm_messages, caller="_generate_doctor_turn")
        except Exception as e:
            logger.error(f"[FamilyDialogueGenerator] Doctor turn generation failed: {e}")
            return "Based on the situation you describe, I will give you some suggestions..."

    async def _extract_session_summary(
        self,
        role: FamilyRole,
        consultation_focus: str,
        messages: List[FamilyNoiseMessage],
    ) -> str:
        """Extract session summary."""
        dialogue_text = "\n".join([
            f"{'User' if m.agent_type == 'user_agent' else 'doctor'}: {m.content}"
            for m in messages
        ])

        prompt = EXTRACT_SUMMARY_PROMPT.format(
            family_relationship=role.relationship,
            family_name=role.name,
            consultation_focus=consultation_focus,
            dialogue_text=dialogue_text,
        )

        try:
            return self._call_llm(
                [{"role": "user", "content": prompt}],
                caller="_extract_session_summary",
                max_tokens=800,
            )
        except Exception as e:
            logger.warning(f"[FamilyDialogueGenerator] Summary extraction failed: {e}")
            return f"关于{role.name}的{consultation_focus}咨询"

    async def _extract_knowledge_points(
        self,
        role: FamilyRole,
        consultation_focus: str,
        messages: List[FamilyNoiseMessage],
        noise_family_id: int,
        past_knowledge_points: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract knowledge points from dialogue.

        Args:
            role: Family role.
            consultation_focus: Consultation focus topic.
            messages: Dialogue message list.
            noise_family_id: Noise ID.
            past_knowledge_points: Historical knowledge points for this role.

        Returns:
            List of knowledge points (1-3 items).
        """
        dialogue_text = "\n".join([
            f"{'User' if m.agent_type == 'user_agent' else 'doctor'}: {m.content}"
            for m in messages
        ])

        # Build historical knowledge points summary for this role
        past_kp_text = ""
        if past_knowledge_points:
            past_kp_summary = "\n".join([
                f"- [{kp.get('category', '')}] {kp.get('name', '')}: {kp.get('content', '')}"
                for kp in past_knowledge_points[-10:]
            ])
            past_kp_text = f"""
## 关于{role.name}的历史知识点记录（参考，避免重复）
{past_kp_summary}
"""

        prompt = f"""请从以下关于亲人健康咨询的对话中提取1-3个关键知识点。
{past_kp_text}
## 亲人信息
- 关系：{role.relationship}
- 称呼：{role.name}
- 年龄：{role.age_range}

## 本次咨询重点
{consultation_focus}

## 对话内容
{dialogue_text}

## 要求
1. 提取1-3个本次对话的核心知识点（必须至少提取1个）
2. 避免与该亲人的历史知识点重复
3. 每个知识点包含：
   - category: 分类（如"用药指导"、"生活调理"、"复查提醒"、"症状观察"等）
   - name: 知识点名称（2-8个字）
   - content: 具体内容摘要（一句话，要具体实用）

请以JSON格式返回数组，只返回JSON数组，不要其他内容。
"""

        try:
            content = self._call_llm(
                [{"role": "user", "content": prompt}],
                caller="_extract_knowledge_points",
                max_tokens=1000,
            )

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            kps = json.loads(content)

            if not isinstance(kps, list):
                kps = [kps]

            if len(kps) > 3:
                kps = kps[:3]

            for kp in kps:
                kp["trap_score"] = 0.1
                kp["noise_family_id"] = noise_family_id
                kp["is_family_noise"] = True
                kp["family_role"] = role.name

            if len(kps) == 0:
                kps = [{
                    "category": "family health",
                    "name": f"{role.name}咨询",
                    "content": f"关于{role.name}的{consultation_focus}",
                    "trap_score": 0.1,
                    "noise_family_id": noise_family_id,
                    "is_family_noise": True,
                    "family_role": role.name,
                }]

            return kps
        except Exception as e:
            logger.warning(f"[FamilyDialogueGenerator] Knowledge points extraction failed: {e}")
            return [{
                "category": "family health",
                "name": f"{role.name}咨询",
                "content": f"关于{role.name}的{consultation_focus}",
                "trap_score": 0.1,
                "noise_family_id": noise_family_id,
                "is_family_noise": True,
                "family_role": role.name,
            }]

    async def generate_session(
        self,
        noise_family_id: int,
        persona: Dict[str, Any],
        role: FamilyRole,
        session_idx: int,
    ) -> FamilyNoiseSession:
        """Generate a single family consultation noise session.

        Args:
            noise_family_id: Noise ID.
            persona: User persona.
            role: Family role.
            session_idx: Session index for this role (for continuity).

        Returns:
            Generated noise session.
        """
        # Select consultation focus
        consultation_focus = await self._select_consultation_focus(role, session_idx)

        logger.info(f"[FamilyDialogueGenerator] Generating session {noise_family_id}: "
                   f"{role.name}({role.relationship}) - {consultation_focus}")

        # Determine dialogue turn count
        num_turns = random.randint(self.config.min_turns, self.config.max_turns)

        messages = []
        for turn in range(1, num_turns + 1):
            # User turn
            user_content = await self._generate_user_turn(
                persona, role, consultation_focus, messages, turn
            )
            messages.append(FamilyNoiseMessage(
                turn=turn,
                role="user",
                content=user_content,
                agent_type="user_agent",
            ))

            # Doctor turn
            doctor_content = await self._generate_doctor_turn(
                role, consultation_focus, messages
            )
            messages.append(FamilyNoiseMessage(
                turn=turn,
                role="assistant",
                content=doctor_content,
                agent_type="doctor_agent",
            ))

        # Get historical knowledge points for this role
        role_key = self._get_role_key(role.persona_id, role.role_id)
        role_past_kps = self._role_knowledge_points.get(role_key, [])

        # Extract session summary and knowledge points
        session_summary = await self._extract_session_summary(role, consultation_focus, messages)
        knowledge_points = await self._extract_knowledge_points(
            role, consultation_focus, messages, noise_family_id, role_past_kps
        )

        # Add newly extracted knowledge points to the role's list
        if role_key not in self._role_knowledge_points:
            self._role_knowledge_points[role_key] = []
        self._role_knowledge_points[role_key].extend(knowledge_points)

        # Record summary for subsequent sessions
        if role_key not in self._role_past_summaries:
            self._role_past_summaries[role_key] = []
        self._role_past_summaries[role_key].append(session_summary)

        session = FamilyNoiseSession(
            noise_family_id=noise_family_id,
            noise_type="family_health_consultation",
            persona_id=role.persona_id,
            family_role=role,
            health_issue=consultation_focus,
            turn_count=num_turns,
            messages=messages,
            knowledge_points=knowledge_points,
            session_summary=session_summary,
            created_at=datetime.now().isoformat(),
        )

        if self.config.verbose:
            logger.info(f"  Complete: {num_turns} turns of dialogue, {len(knowledge_points)} knowledge points")
            logger.info(f"  Role {role.name} cumulative knowledge points: {len(self._role_knowledge_points[role_key])}")

        return session

    async def generate_all_for_persona(
        self,
        persona: Dict[str, Any],
        start_id: int,
    ) -> List[FamilyNoiseSession]:
        """Generate all family consultation sessions for a single user persona.

        Args:
            persona: User persona.
            start_id: Starting noise ID.

        Returns:
            List of generated noise sessions.
        """
        persona_id = persona["persona_id"]
        logger.info(f"[FamilyDialogueGenerator] Starting family consultation session generation for persona {persona_id}")

        # Generate family roles first (with detailed health conditions)
        roles = await self.generate_family_roles(persona)

        sessions = []
        current_id = start_id

        # Generate sessions for each role
        for role in roles:
            logger.info(f"  Processing role: {role.name}({role.relationship}), "
                       f"{len(role.health_conditions)} health questions")
            for session_idx in range(self.config.sessions_per_role):
                session = await self.generate_session(
                    noise_family_id=current_id,
                    persona=persona,
                    role=role,
                    session_idx=session_idx,
                )
                sessions.append(session)
                current_id += 1

        logger.info(f"  Persona {persona_id} complete, total {len(sessions)} sessions")
        return sessions

    async def generate_all(self) -> List[FamilyNoiseSession]:
        """Generate all family/friends noise sessions.

        Returns:
            List of generated noise sessions.
        """
        # Load user personas
        personas = self.load_personas()

        all_sessions = []
        current_id = 1

        for persona in personas:
            sessions = await self.generate_all_for_persona(persona, current_id)
            all_sessions.extend(sessions)
            current_id += len(sessions)

        logger.info(f"[FamilyDialogueGenerator] Generation complete, total {len(all_sessions)} noise sessions")
        return all_sessions

    def save_sessions(
        self,
        sessions: List[FamilyNoiseSession],
        output_path: str,
    ) -> None:
        """Save noise sessions to file."""
        data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_noise_sessions": len(sessions),
                "total_turns": sum(s.turn_count for s in sessions),
                "noise_type": "family_health_consultation",
            },
            "noise_sessions": [s.to_dict() for s in sessions],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[FamilyDialogueGenerator] Noise sessions saved to: {output_path}")

    def save_roles(self, output_path: str) -> None:
        """Save generated family roles to file."""
        all_roles = []
        for persona_id, roles in self._roles_by_persona.items():
            for role in roles:
                all_roles.append(role.to_dict())

        data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_roles": len(all_roles),
            },
            "family_roles": all_roles,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[FamilyDialogueGenerator] Family roles saved to: {output_path}")
