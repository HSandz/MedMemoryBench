"""Noise session generator.

Generates health-knowledge chitchat noise session data.
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

from .config import NoiseConfig

logger = logging.getLogger(__name__)


# Noise user agent system prompt
NOISE_USER_SYSTEM_PROMPT = 'You are an ordinary user who is consulting with an AI doctor through an online medical and health consultation platform.\n\n## Consultation type\nYour inquiry may fall into one of the following categories:\n1. **Symptom consultation**: Describe the uncomfortable symptoms corresponding to certain diseases, and want to know the possible medical causes and coping methods (the most important type)\n2. **Medical knowledge**: Ask about some medical knowledge, such as when you need medical treatment, medication precautions, etc.\n3. **Preventive Care**: Learn how to prevent certain diseases or maintain a healthy lifestyle\n4. **Health Questions**: I have questions about the health information I saw online and want to ask for confirmation or learn more.\n\n## Current consultation topics\n{topic}\n\n## Consulting background\n{background}\n\n## Conversation style requirements\n- Express yourself as naturally as a real patient/consultant\n- If it is a symptom consultation, describe the symptom effect, duration, accompanying symptoms and other details, but do not use personal pronouns such as "I" or "family", just like simply describing the medical symptom consultation question\n- If it is a common sense consultation, you can explain why you want to know about this issue\n- You can ask for details or express concerns based on the doctor’s answer\n- The tone is natural and can be somewhat colloquial\n-Reply length is moderate, 1-4 sentences\n\n## Notes\n- This is an independent health consultation and has nothing to do with your previous medical visits\n- Don\'t mention anything like "you\'ve seen a doctor before" or "the last time the doctor said" when answering.\n- Focus on the current topic of inquiry\n- To avoid duplication of current consultation topics and previous topics, consult on different content\n\n## Past consultation (try to avoid building the same content repeatedly)\n{past_summary}'

# Noise doctor agent system prompt
NOISE_DOCTOR_SYSTEM_PROMPT = """You are a professional and patient AI doctor who is providing online health consultation services to users.

## Role positioning
- Professional and rigorous yet easy-to-understand language
- Warm and patient, good at listening
- Focus on health education and prevention

## Reply Principle
1. **Consultation based on symptoms**:
   - Ask for necessary details (duration, accompanying symptoms, aggravating/mitigating factors, etc.)
   - Analyze possible causes but avoid making a direct diagnosis
   - Give preliminary response suggestions
   - It is recommended to seek medical examination if necessary

2. **Based on medical common sense**:
   - Provide accurate and practical medical knowledge
   -Explain professional concepts in layman’s terms
   - Correct common health misunderstandings
   - Emphasize individual differences and recommend specific consultation on specific issues

3. **For preventive care**:
   - Provide scientific prevention advice
   - Recommend healthy lifestyle
   - Explain the rationale behind preventive measures

## Reply to request
- The reply length is moderate, like an online consultation with a real doctor
- You can explain in points, but don’t be too dogmatic.
- Express concern appropriately
- Important reminder: If you have severe symptoms, you should seek medical treatment promptly."""


@dataclass
class NoiseMessage:
    """A single message within a noise session."""
    turn: int
    role: str  # "user" or "assistant"
    content: str
    agent_type: str  # "user_agent" or "doctor_agent"


@dataclass
class NoiseSession:
    """A single noise session."""
    noise_id: int
    noise_type: str = "health_knowledge"
    topic: str = ""
    turn_count: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_points: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        # Messages may be NoiseMessage objects or dicts
        messages_list = []
        for msg in self.messages:
            if isinstance(msg, NoiseMessage):
                messages_list.append({
                    "turn": msg.turn,
                    "role": msg.role,
                    "content": msg.content,
                    "agent_type": msg.agent_type,
                })
            else:
                messages_list.append(msg)
        
        return {
            "noise_id": self.noise_id,
            "noise_type": self.noise_type,
            "topic": self.topic,
            "turn_count": self.turn_count,
            "messages": messages_list,
            "knowledge_points": self.knowledge_points,
            "created_at": self.created_at,
        }


class NoiseDialogueGenerator:
    """Noise session generator.

    Generates health-knowledge chitchat noise sessions for testing agent memory robustness.
    """

    def __init__(self, config: Optional[NoiseConfig] = None):
        """Initialize the generator.

        Args:
            config: Noise data configuration.
        """
        self.config = config or NoiseConfig()

        # Load API config
        self._load_api_config()

        # LLM client (lazy initialization)
        self._client = None

        # Generated session records (for deduplication)
        self._generated_topics: List[str] = []
        self._past_summaries: List[str] = []

        # Global knowledge points list (accumulated across sessions)
        self._all_knowledge_points: List[Dict[str, Any]] = []

        logger.info("[NoiseDialogueGenerator] Initialization complete")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Planned sessions to generate: {self.config.num_noise_sessions}")

    def _load_api_config(self) -> None:
        """Load API config."""
        import os
        from dotenv import load_dotenv

        # Load .env file
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        self.api_key = os.getenv("OPENAI_API_KEY")
        # Support both OPENAI_API_BASE and OPENAI_BASE_URL
        self.base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        # Support both LLM_MODEL and DEFAULT_LLM_MODEL; strip litellm "openai/" prefix
        model = self.config.model or os.getenv("LLM_MODEL") or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        # Strip "openai/" prefix if present (litellm format to OpenAI format)
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
        """Get completion parameters compatible with different models.

        Newer OpenAI API uses max_completion_tokens instead of max_tokens.
        Defaults to max_completion_tokens (new standard); uses max_tokens only for known legacy models.

        Args:
            max_tokens: Maximum number of tokens.

        Returns:
            Dict with the correct parameter name.
        """
        # Known legacy model prefixes that only support max_tokens
        legacy_model_prefixes = (
            "gpt-3.5",
            "gpt-4-",  # gpt-4-turbo and other legacy versions
            "text-davinci",
            "text-curie",
            "text-babbage",
            "text-ada",
        )

        # Use max_tokens for legacy models
        if any(self.model.startswith(prefix) for prefix in legacy_model_prefixes):
            return {"max_tokens": max_tokens}
        else:
            # Default to new standard: max_completion_tokens
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
            max_tokens: Maximum tokens (defaults to config value).

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
                caller=f"noise.{caller}",
            )

        return response.choices[0].message.content.strip()

    def _select_topic(self) -> str:
        """Select a consultation topic (avoiding duplicates)."""
        available_topics = [
            t for t in self.config.health_topics
            if t not in self._generated_topics
        ]

        if not available_topics:
            # All topics used; allow repeats with variants
            base_topic = random.choice(self.config.health_topics)
            variants = [
                f"{base_topic}（深入讨论）",
                f"{base_topic}（实践建议）",
                f"{base_topic}（常见误区）",
                f"{base_topic}（科学解读）",
            ]
            return random.choice(variants)

        topic = random.choice(available_topics)
        self._generated_topics.append(topic)
        return topic

    def _generate_background(self, topic: str) -> str:
        """Generate a consultation background for the topic.

        Produces different backgrounds based on topic type for more natural dialogues.
        """
        # Backgrounds for symptom-related topics
        symptom_backgrounds = [
            f"你最近出现了{topic}的情况，持续了好几天，有点担心想咨询一下。",
            f"你发现自己{topic}，不知道是什么原因，想了解一下要不要去医院看看。",
            f"你家人出现了{topic}的症状，你帮他来咨询一下是怎么回事。",
            f"你{topic}已经有一段Time了，之前没太在意，最近感觉有点加重了。",
            f"你今天突然{topic}，有些担心，想先问问医生的意见。",
            f"你工作比较忙，{topic}的Question一直拖着没去看，想先在线咨询一下。",
        ]

        # Backgrounds for general knowledge topics
        knowledge_backgrounds = [
            f"你一直搞不清楚{topic}，今天想彻底弄明白。",
            f"你的friend问过你{topic}，你也不太确定，想找专业人士确认一下。",
            f"你在网上看到关于{topic}的说法众说纷纭，想听听医生的专业意见。",
            f"你最近在关注健康知识，对{topic}特别想了解清楚。",
            f"你家里老人问你{topic}，你想先了解清楚再告诉他们。",
            f"你马上要做相关检查了，想提前了解一下{topic}。",
        ]

        # Backgrounds for prevention/wellness topics
        prevention_backgrounds = [
            f"你想了解一下{topic}，做好日常预防。",
            f"你身边有人出现相关Question，你想提前了解{topic}。",
            f"你最近Start注重健康管理，想系统了解一下{topic}。",
            f"换季了，你想提前了解{topic}的注意事项。",
            f"你觉得自己的生活习惯可能不太健康，想咨询一下{topic}。",
            f"你年纪渐长，想了解一下{topic}方面的知识。",
        ]

        # Determine topic type by keywords and select background
        symptom_keywords = ["pain", "itch", "numb", "swell", "stuffy", "faint", "cough", "pain", "discomfort", "Foreign body sensation",
                           "Insomnia", "fatigue", "Lack of energy", "ulcer", "tinnitus", "Vague", "Flustered"]
        knowledge_keywords = ["What's going on", "how many degrees", "How to see", "normal value", "Things to note", "Taboo",
                             "can be together", "how long", "need"]

        if any(kw in topic for kw in symptom_keywords):
            return random.choice(symptom_backgrounds)
        elif any(kw in topic for kw in knowledge_keywords):
            return random.choice(knowledge_backgrounds)
        else:
            return random.choice(prevention_backgrounds)

    def _get_past_summary(self) -> str:
        """Get past consultation summaries (for deduplication)."""
        if not self._past_summaries:
            return "(This is your first consultation)"

        # Show only the most recent entries
        recent = self._past_summaries[-5:]
        return "You have previously consulted on the following topics:" + "\n".join(f"- {s}" for s in recent)

    async def _generate_user_turn(
        self,
        topic: str,
        background: str,
        past_summary: str,
        messages: List[Dict[str, Any]],
        turn: int,
    ) -> str:
        """Generate a user turn.

        Args:
            topic: Consultation topic.
            background: Consultation background.
            past_summary: Past consultation summary.
            messages: Current dialogue history.
            turn: Current turn number.

        Returns:
            User reply content.
        """
        system_prompt = NOISE_USER_SYSTEM_PROMPT.format(
            topic=topic,
            background=background,
            past_summary=past_summary,
        )

        llm_messages = [{"role": "system", "content": system_prompt}]

        # Add dialogue history
        for msg in messages:
            role = "assistant" if msg["agent_type"] == "user_agent" else "user"
            llm_messages.append({"role": role, "content": msg["content"]})

        # First turn prompt
        if turn == 1:
            llm_messages.append({
                "role": "user",
                "content": f"请根据你的背景和话题，向医生提出你的第一个Question。"
            })

        try:
            return self._call_llm(llm_messages, caller="_generate_user_turn")
        except Exception as e:
            logger.error(f"[NoiseDialogueGenerator] User turn generation failed: {e}")
            return f"我想了解一下{topic}方面的知识。"

    async def _generate_doctor_turn(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Generate a doctor turn.

        Args:
            messages: Current dialogue history.

        Returns:
            Doctor reply content.
        """
        llm_messages = [{"role": "system", "content": NOISE_DOCTOR_SYSTEM_PROMPT}]

        # Add dialogue history
        for msg in messages:
            role = "user" if msg["agent_type"] == "user_agent" else "assistant"
            llm_messages.append({"role": role, "content": msg["content"]})

        try:
            return self._call_llm(llm_messages, caller="_generate_doctor_turn")
        except Exception as e:
            logger.error(f"[NoiseDialogueGenerator] Doctor turn generation failed: {e}")
            return "This is a good question. Let me answer it for you..."

    async def _extract_knowledge_points(
        self,
        topic: str,
        messages: List[Dict[str, Any]],
        noise_id: int,
        past_knowledge_points: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract knowledge point summaries from the dialogue.

        Args:
            topic: Consultation topic.
            messages: Dialogue message list.
            noise_id: Noise session ID.
            past_knowledge_points: Historical knowledge points list (for reference and deduplication).

        Returns:
            List of knowledge points (1-3 items).
        """
        dialogue_text = "\n".join([
            f"{'User' if m['agent_type'] == 'user_agent' else 'doctor'}: {m['content']}"
            for m in messages
        ])

        # Build historical knowledge points summary
        past_kp_text = ""
        if past_knowledge_points:
            past_kp_summary = "\n".join([
                f"- [{kp.get('category', '')}] {kp.get('name', '')}: {kp.get('content', '')}"
                for kp in past_knowledge_points[-10:]  # Show only last 10
            ])
            past_kp_text = f"""
## 历史知识点记录（参考，避免重复）
{past_kp_summary}
"""

        prompt = f"""请从以下健康知识咨询对话中提取1-3个关键知识点。
{past_kp_text}
## 本次咨询话题
{topic}

## 对话内容
{dialogue_text}

## 要求
1. 提取1-3个本次对话的核心知识点（必须至少提取1个）
2. 避免与历史知识点重复
3. 每个知识点包含：
   - category: 分类（如"健康知识"、"预防建议"、"生活方式"等）
   - name: 知识点名称（2-6个字）
   - content: 具体内容摘要（一句话）

请以JSON格式返回数组，只返回JSON数组，不要其他内容。
"""

        try:
            content = self._call_llm(
                [{"role": "user", "content": prompt}],
                caller="_extract_knowledge_points",
                max_tokens=1000,
            )

            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            kps = json.loads(content)

            # Ensure result is a list
            if not isinstance(kps, list):
                kps = [kps]

            # Limit to 1-3 items
            if len(kps) > 3:
                kps = kps[:3]

            # Add extra fields
            for kp in kps:
                kp["trap_score"] = 0.1  # Low score for noise knowledge points
                kp["noise_id"] = noise_id
                kp["is_noise"] = True

            # Ensure at least 1 entry
            if len(kps) == 0:
                kps = [{
                    "category": "health knowledge",
                    "name": topic[:6],
                    "content": f"关于{topic}的咨询记录",
                    "trap_score": 0.1,
                    "noise_id": noise_id,
                    "is_noise": True,
                }]

            return kps
        except Exception as e:
            logger.warning(f"[NoiseDialogueGenerator] Knowledge point extraction failed: {e}")
            return [{
                "category": "health knowledge",
                "name": topic[:6],
                "content": f"关于{topic}的咨询记录",
                "trap_score": 0.1,
                "noise_id": noise_id,
                "is_noise": True,
            }]

    async def generate_session(self, noise_id: int) -> NoiseSession:
        """Generate a single noise session.

        Args:
            noise_id: Noise session ID.

        Returns:
            The generated noise session.
        """
        # Select topic and background
        topic = self._select_topic()
        background = self._generate_background(topic)
        past_summary = self._get_past_summary()

        logger.info(f"[NoiseDialogueGenerator] Generating noise session {noise_id}: {topic}")

        # Determine number of dialogue turns
        num_turns = random.randint(self.config.min_turns, self.config.max_turns)

        messages = []
        for turn in range(1, num_turns + 1):
            # User turn
            user_content = await self._generate_user_turn(
                topic, background, past_summary, messages, turn
            )
            messages.append({
                "turn": turn,
                "role": "user",
                "content": user_content,
                "agent_type": "user_agent",
            })

            # Doctor turn
            doctor_content = await self._generate_doctor_turn(messages)
            messages.append({
                "turn": turn,
                "role": "assistant",
                "content": doctor_content,
                "agent_type": "doctor_agent",
            })

        # Extract knowledge points (pass historical list for deduplication)
        knowledge_points = await self._extract_knowledge_points(
            topic, messages, noise_id, self._all_knowledge_points
        )

        # Add newly extracted knowledge points to the global list
        self._all_knowledge_points.extend(knowledge_points)

        # Record summary for subsequent deduplication
        self._past_summaries.append(f"{topic}")

        session = NoiseSession(
            noise_id=noise_id,
            noise_type="health_knowledge",
            topic=topic,
            turn_count=num_turns,
            messages=messages,
            knowledge_points=knowledge_points,
            created_at=datetime.now().isoformat(),
        )

        if self.config.verbose:
            logger.info(f"  Done: {num_turns} turns, {len(knowledge_points)} knowledge points")
            logger.info(f"  Global knowledge points total: {len(self._all_knowledge_points)}")

        return session

    async def generate_all(self) -> List[NoiseSession]:
        """Generate all noise sessions.

        Returns:
            List of generated noise sessions.
        """
        logger.info(f"[NoiseDialogueGenerator] Starting generation of {self.config.num_noise_sessions} noise sessions")

        sessions = []
        for i in range(self.config.num_noise_sessions):
            noise_id = i + 1
            session = await self.generate_session(noise_id)
            sessions.append(session)

        logger.info(f"[NoiseDialogueGenerator] Generation complete, total {len(sessions)} noise sessions")
        return sessions

    def save_sessions(
        self,
        sessions: List[NoiseSession],
        output_path: str,
    ) -> None:
        """Save noise sessions to file.

        Args:
            sessions: List of noise sessions.
            output_path: Output file path.
        """
        data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_noise_sessions": len(sessions),
                "total_turns": sum(s.turn_count for s in sessions),
                "noise_type": "health_knowledge",
            },
            "noise_sessions": [s.to_dict() for s in sessions],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[NoiseDialogueGenerator] Noise sessions saved to: {output_path}")
