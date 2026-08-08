"""Noise data augmentation configuration."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class NoiseConfig:
    """Noise data augmentation configuration."""

    data_dir: str = "data"
    input_filename: str = "generated_dialogues.json"
    output_filename: str = "generated_dialogues_with_noise.json"

    num_noise_sessions: int = 20
    min_turns: int = 3
    max_turns: int = 8

    model: Optional[str] = None
    temperature: float = 1.0
    max_tokens: int = 1500

    health_topics: List[str] = field(default_factory=lambda: [
        "I have a foreign body sensation in my throat in the morning and I can’t cough it out.",
        "Persistent headache but no fever",
        "Often feel chest tightness and shortness of breath",
        "I've been having insomnia and dreaming lately",
        "Stomach bloating and indigestion after eating",
        "The skin suddenly breaks out in a rash and is itchy",
        "Dry eyes and blurred vision",
        "Back pain, uncomfortable sitting for long periods of time",
        "What causes numbness in hands and feet?",
        "Often feel tired and lack of energy",
        "Oral ulcers recurring",
        "tinnitus hearing loss",
        "Joint pain and limited movement",
        "Constipation or diarrhea alternating with each other",
        "Rapid heartbeat, palpitation",
        "When do you need a shot for a cold?",
        "At what level of fever do you need to take antipyretics?",
        "When should you take antibiotics?",
        "How to read the physical examination report",
        "What are the normal values ​​for blood pressure and blood sugar?",
        "Precautions for vaccination",
        "What are the contraindications between drugs?",
        "Can I take Chinese and Western medicine together?",
        "How long will it take to be able to carry out normal activities after surgery?",
        "Do chronic diseases require long-term medication?",
        "How to Prevent Seasonal Flu",
        "How to protect your cervical spine when sitting for long periods of time in the office",
        "What harm does staying up late do to your body?",
        "How to improve immunity",
        "How to make a healthier diet",
        "Exercise methods suitable for middle-aged and elderly people",
        "How to protect eyes and prevent myopia",
        "How to improve poor sleep quality",
        "How to regulate emotions when stressed",
        "How to prevent three highs",
        "What to pay attention to during pregnancy",
        "Nutritional supplements for children during development",
        "Osteoporosis prevention in the elderly",
        "Things to note during women’s menstrual period",
        "Youth mental health issues",
        "Precautions for postpartum recovery",
        "Coping with menopausal syndrome",
        "Nursing care for common diseases in infants and young children",
        "How to eat healthily during weight loss",
        "What are the best ways to quit smoking and drinking?",
        "How to deal with sports injuries",
        "The health effects of looking at mobile phones for a long time",
        "Is it necessary to take health supplements?",
        "Basic principles of traditional Chinese medicine health care",
        "Under what circumstances do you need to go to the hospital for examination?",
        "What are the household medicines?",
    ])

    verbose: bool = True
    dry_run: bool = False
