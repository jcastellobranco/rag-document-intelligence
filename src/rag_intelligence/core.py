from collections import Counter
from dataclasses import dataclass
import math
import re

def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

@dataclass(frozen=True)
class Passage:
    source: str
    index: int
    text: str

class Index:
    def __init__(self):
        self.passages: list[Passage] = []

    def add(self, source: str, text: str, chunk_words: int = 120, overlap: int = 20) -> int:
        if chunk_words <= 0 or not 0 <= overlap < chunk_words:
            raise ValueError("Invalid chunking parameters")
        words = text.split()
        count = 0
        for start in range(0, len(words), chunk_words - overlap):
            chunk = words[start:start + chunk_words]
            if not chunk:
                break
            self.passages.append(Passage(source, count, " ".join(chunk)))
            count += 1
            if start + chunk_words >= len(words):
                break
        return count

    def search(self, question: str, limit: int = 3) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query = Counter(tokens(question))
        if not query:
            return []
        docs = [Counter(tokens(p.text)) for p in self.passages]
        n = len(docs)
        df = Counter(word for doc in docs for word in doc)
        def vector(doc):
            return {word: (1 + math.log(freq)) * math.log((n + 1) / (df[word] + 1) + 1) for word, freq in doc.items()}
        q = vector(query)
        qnorm = math.sqrt(sum(x*x for x in q.values()))
        scored = []
        for passage, counts in zip(self.passages, docs):
            d = vector(counts)
            norm = math.sqrt(sum(x*x for x in d.values()))
            score = sum(q.get(word, 0) * val for word, val in d.items()) / (qnorm * norm or 1)
            if score > 0:
                scored.append({"source": passage.source, "chunk": passage.index, "text": passage.text, "score": round(score, 5)})
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:limit]
