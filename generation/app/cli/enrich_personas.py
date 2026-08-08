#!/usr/bin/env python3
"""CLI tool for enriching user personas using LLM.

Usage:
    python -m app.cli.enrich_personas [OPTIONS]

Examples:
    python -m app.cli.enrich_personas
    python -m app.cli.enrich_personas --ids 1,2,3
    python -m app.cli.enrich_personas --dry-run
    python -m app.cli.enrich_personas --input data/personas.json --output data/enriched.json
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import get_settings
from app.prompts.persona_enrich import build_enrich_prompt
from app.schemas.persona_enriched import (
    BasePersonaInput,
    BatchEnrichmentResult,
    EnrichedFieldsSchema,
    EnrichedPersonaSchema,
    EnrichmentResult,
)
from app.services.llm import LLMService

TEMPERATURE = 1.0


class PersonaEnricher:
    """Service for enriching user personas using LLM."""

    def __init__(self, model: Optional[str] = None):
        self.settings = get_settings()
        self.llm = LLMService(model=model)
        self.max_retries = 3

    async def enrich_single(
        self, persona: dict, retry_count: int = 0
    ) -> EnrichmentResult:
        """Enrich a single persona with LLM-generated background details."""
        persona_id = persona["id"]

        try:
            BasePersonaInput(**persona)

            prompt = build_enrich_prompt(persona)

            messages = [
                {
                    "role": "system",
                    "content": "You are a professional user portrait expansion assistant. Please strictly follow the required JSON format output.",
                },
                {"role": "user", "content": prompt},
            ]

            response = await self.llm.complete_json(
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=2000,
                caller="cli.enrich_personas",
            )

            enriched_fields = EnrichedFieldsSchema(**response)

            enriched_persona = EnrichedPersonaSchema(
                id=persona["id"],
                type_name=persona["type_name"],
                gender=persona["gender"],
                core_feature=persona["core_feature"],
                health_goals=persona["health_goals"],
                category=persona["category"],
                enriched=enriched_fields,
            )

            return EnrichmentResult(
                success=True,
                persona_id=persona_id,
                enriched_persona=enriched_persona,
            )

        except ValidationError as e:
            error_msg = f"Validation error: {e}"
            if retry_count < self.max_retries:
                print(f"  [!] Retry {retry_count + 1}/{self.max_retries} for persona {persona_id}")
                return await self.enrich_single(persona, retry_count + 1)
            return EnrichmentResult(
                success=False,
                persona_id=persona_id,
                error=error_msg,
            )

        except json.JSONDecodeError as e:
            error_msg = f"JSON parse error: {e}"
            if retry_count < self.max_retries:
                print(f"  [!] Retry {retry_count + 1}/{self.max_retries} for persona {persona_id}")
                return await self.enrich_single(persona, retry_count + 1)
            return EnrichmentResult(
                success=False,
                persona_id=persona_id,
                error=error_msg,
            )

        except Exception as e:
            return EnrichmentResult(
                success=False,
                persona_id=persona_id,
                error=str(e),
            )

    async def enrich_batch(
        self,
        personas: list[dict],
        progress_callback: Optional[callable] = None,
    ) -> BatchEnrichmentResult:
        """Enrich a batch of personas sequentially."""
        results = []
        failed_ids = []
        total = len(personas)

        for i, persona in enumerate(personas):
            if progress_callback:
                progress_callback(i + 1, total, persona["id"])

            result = await self.enrich_single(persona)
            results.append(result)

            if not result.success:
                failed_ids.append(result.persona_id)

        success_count = sum(1 for r in results if r.success)

        return BatchEnrichmentResult(
            total=total,
            success_count=success_count,
            failure_count=total - success_count,
            results=results,
            failed_ids=failed_ids,
        )


def load_personas(input_path: Path) -> list[dict]:
    """Load personas from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_enriched_personas(
    results: list[EnrichmentResult],
    output_path: Path,
) -> None:
    """Save enriched personas to JSON file."""
    enriched = []
    for result in results:
        if result.success and result.enriched_persona:
            enriched.append(result.enriched_persona.model_dump())

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)


def print_progress(current: int, total: int, persona_id: int) -> None:
    """Print progress to console."""
    percent = (current / total) * 100
    print(f"[{current}/{total}] ({percent:.1f}%) Processing persona {persona_id}...")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    project_root = Path(__file__).parent.parent.parent.parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    print(f"Loading personas from: {input_path}")
    try:
        personas = load_personas(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        return 1

    print(f"Loaded {len(personas)} personas")

    if args.ids:
        id_list = [int(x.strip()) for x in args.ids.split(",")]
        personas = [p for p in personas if p["id"] in id_list]
        print(f"Filtered to {len(personas)} personas (IDs: {id_list})")

    if not personas:
        print("No personas to process")
        return 0

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print("Would process the following personas:")
        for p in personas:
            print(f"  - ID {p['id']}: {p['type_name']} ({p['gender']}, {p['core_feature'][:30]}...)")
        print(f"\nOutput would be saved to: {output_path}")
        return 0

    enricher = PersonaEnricher(model=args.model)

    print(f"\nStarting enrichment (model: {enricher.llm.model})...")
    print("-" * 50)

    start_time = time.time()
    result = await enricher.enrich_batch(personas, progress_callback=print_progress)
    elapsed = time.time() - start_time

    print("-" * 50)
    print(f"\nEnrichment completed in {format_duration(elapsed)}")
    print(f"  Total: {result.total}")
    print(f"  Success: {result.success_count}")
    print(f"  Failed: {result.failure_count}")

    if result.failed_ids:
        print(f"  Failed IDs: {result.failed_ids}")
        print("\nError details:")
        for r in result.results:
            if not r.success:
                print(f"  - Persona {r.persona_id}: {r.error}")

    if result.success_count > 0:
        save_enriched_personas(result.results, output_path)
        print(f"\nSaved {result.success_count} enriched personas to: {output_path}")

    return 0 if result.failure_count == 0 else 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enrich user personas using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        default="user_personas.json",
        help="Input file path (default: user_personas.json)",
    )
    parser.add_argument(
        "--output",
        default="user_personas_enriched.json",
        help="Output file path (default: user_personas_enriched.json)",
    )
    parser.add_argument(
        "--ids",
        help="Comma-separated list of persona IDs to process (default: all)",
    )
    parser.add_argument(
        "--model",
        help="LLM model to use (default: from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without calling LLM",
    )

    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
