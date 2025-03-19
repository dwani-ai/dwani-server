from typing import List

def chunk_text(text: str, chunk_size: int = 10) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]