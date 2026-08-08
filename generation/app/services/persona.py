"""Persona service for handling persona operations."""
import json
from typing import Optional
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import BasePersona, ExpandedPersona
from ..schemas.persona import (
    ExpandedPersonaCreate,
    ExpandedPersonaUpdate,
    PersonaExpandRequest,
)
from ..schemas.persona_enriched import EnrichedFieldsSchema
from ..prompts.persona_enrich import build_enrich_prompt
from .llm import get_llm_service

# Default LLM hyperparameters
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 2000


class PersonaService:
    """Service for persona operations."""

    def __init__(
        self,
        db: AsyncSession,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.db = db
        self.llm = get_llm_service()
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def get_all_base_personas(self) -> list[BasePersona]:
        """Get all base personas."""
        result = await self.db.execute(select(BasePersona))
        return list(result.scalars().all())

    async def get_base_persona(self, persona_id: int) -> Optional[BasePersona]:
        """Get a single base persona by ID."""
        result = await self.db.execute(
            select(BasePersona).where(BasePersona.id == persona_id)
        )
        return result.scalar_one_or_none()

    async def get_all_expanded_personas(self) -> list[ExpandedPersona]:
        """Get all expanded personas with their base personas."""
        result = await self.db.execute(
            select(ExpandedPersona).options(selectinload(ExpandedPersona.base_persona))
        )
        return list(result.scalars().all())

    async def get_expanded_persona(self, persona_id: int) -> Optional[ExpandedPersona]:
        """Get a single expanded persona by ID."""
        result = await self.db.execute(
            select(ExpandedPersona)
            .where(ExpandedPersona.id == persona_id)
            .options(selectinload(ExpandedPersona.base_persona))
        )
        return result.scalar_one_or_none()

    async def create_expanded_persona(
        self, data: ExpandedPersonaCreate
    ) -> ExpandedPersona:
        """Create a new expanded persona manually."""
        persona = ExpandedPersona(
            base_persona_id=data.base_persona_id,
            enriched_data=data.enriched_data.model_dump() if data.enriched_data else None,
        )
        self.db.add(persona)
        await self.db.flush()
        await self.db.refresh(persona)
        return persona

    async def update_expanded_persona(
        self, persona_id: int, data: ExpandedPersonaUpdate
    ) -> Optional[ExpandedPersona]:
        """Update an expanded persona."""
        persona = await self.get_expanded_persona(persona_id)
        if not persona:
            return None

        if data.enriched_data is not None:
            persona.enriched_data = data.enriched_data.model_dump()

        await self.db.flush()
        await self.db.refresh(persona)
        return persona

    async def delete_expanded_persona(self, persona_id: int) -> bool:
        """Delete an expanded persona."""
        persona = await self.get_expanded_persona(persona_id)
        if not persona:
            return False

        await self.db.delete(persona)
        return True

    async def expand_persona(
        self, request: PersonaExpandRequest, max_retries: int = 3
    ) -> ExpandedPersona:
        """Expand a base persona using LLM.

        Uses the enriched prompt from persona_enrich.py for structured output.

        Args:
            request: PersonaExpandRequest with base_persona_id
            max_retries: Maximum retry attempts for validation errors

        Returns:
            ExpandedPersona with enriched_data
        """
        base_persona = await self.get_base_persona(request.base_persona_id)
        if not base_persona:
            raise ValueError(f"Base persona {request.base_persona_id} not found")

        # Build persona dict for prompt
        persona_dict = {
            "id": base_persona.id,
            "type_name": base_persona.type_name,
            "gender": base_persona.gender,
            "core_feature": base_persona.core_feature or "",
            "health_goals": base_persona.health_goals or [],
            "category": base_persona.category or "",
        }

        # Build prompt using enrich prompt builder
        prompt = build_enrich_prompt(persona_dict)

        # Call LLM with retry logic
        retry_count = 0
        enriched_fields: Optional[EnrichedFieldsSchema] = None

        while retry_count <= max_retries:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a professional user portrait expansion assistant. Please strictly follow the required JSON format output.",
                    },
                    {"role": "user", "content": prompt},
                ]

                result = await self.llm.complete_json(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    caller="persona.expand_persona",
                )

                # Normalize LLM output to fix common type errors
                result = self._normalize_enriched_result(result)

                # Validate output structure
                enriched_fields = EnrichedFieldsSchema(**result)

                break
            except (ValidationError, json.JSONDecodeError) as e:
                retry_count += 1
                if retry_count > max_retries:
                    raise ValueError(f"Failed to get valid response after {max_retries} retries: {e}")
                continue

        # Create expanded persona with enriched data
        persona = ExpandedPersona(
            base_persona_id=request.base_persona_id,
            enriched_data=enriched_fields.model_dump() if enriched_fields else None,
        )

        self.db.add(persona)
        await self.db.flush()
        await self.db.refresh(persona)

        # Re-fetch with base_persona relationship loaded for serialization
        return await self.get_expanded_persona(persona.id)

    def _normalize_enriched_result(self, result: dict) -> dict:
        """Normalize LLM output, fixing common type errors for lifestyle and health_details."""
        lifestyle = result.get("lifestyle")
        if lifestyle is not None and not isinstance(lifestyle, dict):
            result["lifestyle"] = {
                "sleep_pattern": str(lifestyle) if lifestyle else "",
                "diet_habits": "",
                "exercise_frequency": "",
                "stress_level": "",
            }

        health_details = result.get("health_details")
        if health_details is not None:
            if not isinstance(health_details, dict):
                result["health_details"] = {
                    "medical_history": [],
                }
            else:
                value = health_details.get("medical_history")
                if value is not None and not isinstance(value, list):
                    health_details["medical_history"] = [str(value)] if value else []

        return result


async def import_base_personas(db: AsyncSession, personas_data: list[dict]) -> int:
    """Import base personas from JSON data.

    Args:
        db: Database session.
        personas_data: List of persona dictionaries from user_personas.json.

    Returns:
        Number of personas imported.
    """
    count = 0
    for data in personas_data:
        # Check if already exists
        result = await db.execute(
            select(BasePersona).where(BasePersona.id == data["id"])
        )
        if result.scalar_one_or_none():
            continue

        persona = BasePersona(
            id=data["id"],
            type_name=data["type_name"],
            gender=data["gender"],
            core_feature=data.get("core_feature"),
            health_goals=data.get("health_goals"),
            category=data.get("category"),
        )
        db.add(persona)
        count += 1

    await db.flush()
    return count
