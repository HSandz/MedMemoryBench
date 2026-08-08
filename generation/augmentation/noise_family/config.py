"""Family/friends noise data configuration."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class FamilyNoiseConfig:
    """Family/friends noise data configuration."""

    data_dir: str = "data"
    personas_filename: str = "generated_personas.json"
    input_filename: str = "generated_dialogues.json"
    output_filename: str = "generated_dialogues_with_family_noise.json"

    num_family_roles: int = 5
    sessions_per_role: int = 20
    min_turns: int = 5
    max_turns: int = 8

    model: Optional[str] = None
    temperature: float = 1.0
    max_tokens: int = 2000

    relationship_types: List[str] = field(default_factory=lambda: [
        "Father",
        "Mother",
        "spouse",
        "child",
        "brothers and sisters",
        "grandparents",
        "uncle/aunt",
        "cousins",
        "close friends",
        "colleague",
    ])

    health_issue_categories: List[str] = field(default_factory=lambda: [
        "Daily management of high blood pressure",
        "Diabetes Diet Control",
        "Coronary heart disease medication consultation",
        "arthritis pain relief",
        "Chronic gastritis treatment",
        "Osteoporosis prevention",
        "Abnormal thyroid function",
        "Chronic bronchitis care",
        "Memory loss concerns",
        "Poor sleep quality",
        "Treating constipation problem",
        "Fall prevention measures",
        "Nutritional supplements for the elderly",
        "Coping with hearing loss",
        "Lumbar disc herniation",
        "Prevention and treatment of cervical spondylosis",
        "Fatty Liver Treatment",
        "menopausal syndrome",
        "stress related symptoms",
        "Myopia worsens",
        "gastroesophageal reflux",
        "Migraine attack",
        "allergic rhinitis",
        "Skin problem consultation",
        "Treatment of fever in children",
        "Cough care for children",
        "Diarrhea treatment in children",
        "children nutritional development",
        "Children's vision protection",
        "Postoperative wound care",
        "Exercise guidance during recovery period",
        "Postoperative dietary precautions",
        "Review time schedule",
        "Anxiety relief",
        "Depressive tendencies concern",
        "Insomnia and anxiety regulation",
        "Emotion management advice",
    ])

    verbose: bool = True
    dry_run: bool = False
