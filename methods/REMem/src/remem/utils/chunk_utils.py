from typing import Dict


def make_chunk_content(extract_method: str, chunk: Dict) -> str:
    if extract_method == "openie":
        if "role" in chunk:
            if (
                ": " not in chunk["content"][:20]
            ):  # if the content is not already prefixed with a role, such as person name
                return f"{chunk['role']}: {chunk['content']}"
        return chunk["content"]
    elif extract_method in ["episodic", "episodic_gist"]:
        if "date" in chunk and "role" in chunk:
            return f"Date: {chunk['date']}\n{chunk['role']}: {chunk['content']}"
        return chunk["content"]
    elif extract_method == "message":
        if "date" in chunk and "role" in chunk:
            return f"[{chunk['date']}] {chunk['role']}: {chunk['content']}"
        return chunk["content"]
    elif extract_method == "temporal":
        # For temporal extraction, include date/time context when available
        if "date" in chunk and "role" in chunk:
            return f"Date: {chunk['date']}\n{chunk['role']}: {chunk['content']}"
        elif "date" in chunk:
            return f"Date: {chunk['date']}\n{chunk['content']}"
        return chunk["content"]
    else:
        raise ValueError(f"Unknown extract method: {extract_method}")
