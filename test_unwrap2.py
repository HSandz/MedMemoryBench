import re
text = """Based on the relevant information from the memory store, carefully review the patient's complete medical history and conduct a comprehensive analysis combining information from multiple visits:

Recently when I get up in the morning, I feel a bit nauseated as soon as I move and my chest feels tight, but my blood sugar readings aren’t particularly high. Could these atypical sensations be some lingering effect from that period when my schedule was chaotic and I missed some insulin doses?

[ANSWER REQUIREMENTS] Please thoroughly search through prior memory content and reason by combining multiple historical data points. Your answer should:
1. Clearly list the memory content you are drawing upon
2. Demonstrate a clear reasoning path (from which information to which conclusions)
3. Provide a final comprehensive judgment

Answer:"""

import re
print("Split result:")
print(re.split(
        r"(?is)\n\s*(?:\[answer requirements?\]|answer format(?:\s*\(critical\))?|"
        r"format requirements|critical instructions|answer\s*:)",
        text,
        maxsplit=1,
    ))

