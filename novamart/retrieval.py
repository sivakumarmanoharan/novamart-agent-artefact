"""Document loading, chunking and embedding search over the policy docs.

Design (deliberately simple, see README "Key decisions"):
- One chunk per "## " section of each Markdown file. The docs are short and
  written section-per-topic, so a section is a natural, citable unit.
- Each chunk is embedded with a Gemini embedding model and kept in memory;
  vectors are cached on disk so we only pay for embeddings when docs change.
- Search = cosine similarity, top-k, with a minimum-score cut-off.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from novamart import config, quota

# An embedder takes a list of texts and a task type ("RETRIEVAL_DOCUMENT" or
# "RETRIEVAL_QUERY") and returns one vector per text. Injectable so tests can
# run without an API key.
Embedder = Callable[[list[str], str], np.ndarray]


@dataclass
class Chunk:
    id: str          # citation id, e.g. "returns-and-refunds#electronics"
    doc_title: str   # e.g. "Returns and Refunds Policy"
    section: str     # e.g. "Electronics"
    text: str        # section body


@dataclass
class Hit:
    chunk: Chunk
    score: float


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_chunks(policy_dir: Path = config.POLICY_DIR) -> list[Chunk]:
    """Split every Markdown file into one chunk per '## ' section."""
    chunks = []
    for path in sorted(policy_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
        doc_title = title_match.group(1).strip() if title_match else path.stem
        # re.split with a capture group returns [preamble, h1, body1, h2, body2, ...]
        parts = re.split(r"^## (.+)$", text, flags=re.MULTILINE)
        for heading, body in zip(parts[1::2], parts[2::2]):
            chunks.append(
                Chunk(
                    id=f"{path.stem}#{_slug(heading)}",
                    doc_title=doc_title,
                    section=heading.strip(),
                    text=body.strip(),
                )
            )
    return chunks


def gemini_embedder(client) -> Embedder:
    """Build an Embedder backed by the Gemini embeddings API."""
    from google.genai import types

    def embed(texts: list[str], task_type: str) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), 100):  # API batch limit is 100
            batch = texts[start : start + 100]
            resp = quota.call(
                config.EMBED_MODEL,
                lambda: client.models.embed_content(
                    model=config.EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=config.EMBED_DIM,
                    ),
                ),
            )
            vectors.extend(e.values for e in resp.embeddings)
        return np.array(vectors, dtype=np.float32)

    return embed


def _normalize(m: np.ndarray) -> np.ndarray:
    # Truncated Gemini embeddings (dim < 3072) are not unit length, so we
    # normalise ourselves; after that, a dot product is cosine similarity.
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


class PolicyIndex:
    """In-memory vector index over policy chunks."""

    def __init__(self, chunks: list[Chunk], embed: Embedder, cache_file: Path | None = None):
        self.chunks = chunks
        self.embed = embed
        self.matrix = _normalize(self._embed_documents(cache_file))

    @staticmethod
    def _doc_text(c: Chunk) -> str:
        # Include title and heading so a section like "Eligibility" carries
        # its context ("Price Match Policy") into the embedding.
        return f"{c.doc_title} - {c.section}\n{c.text}"

    def _embed_documents(self, cache_file: Path | None) -> np.ndarray:
        texts = [self._doc_text(c) for c in self.chunks]
        # Cache key covers model, dimension and exact text, so editing a doc
        # or switching models automatically re-embeds.
        key = hashlib.sha256(
            json.dumps([config.EMBED_MODEL, config.EMBED_DIM, texts]).encode()
        ).hexdigest()
        if cache_file and cache_file.exists():
            cached = json.loads(cache_file.read_text())
            if cached.get("key") == key:
                return np.array(cached["vectors"], dtype=np.float32)
        vectors = self.embed(texts, "RETRIEVAL_DOCUMENT")
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"key": key, "vectors": vectors.tolist()}))
        return vectors

    def search(self, query: str, top_k: int = config.TOP_K) -> list[Hit]:
        """Return the top_k chunks by cosine similarity (no threshold applied)."""
        q = _normalize(self.embed([query], "RETRIEVAL_QUERY"))[0]
        scores = self.matrix @ q
        best = np.argsort(-scores)[:top_k]
        return [Hit(self.chunks[i], float(scores[i])) for i in best]


def build_index(client) -> PolicyIndex:
    return PolicyIndex(
        load_chunks(), gemini_embedder(client), cache_file=config.CACHE_DIR / "embeddings.json"
    )


if __name__ == "__main__":
    # Calibration helper: shows raw scores so you can pick MIN_SCORE.
    #   python -m novamart.retrieval "can I return an opened laptop?"
    from novamart.agent import make_client

    index = build_index(make_client())
    for hit in index.search(" ".join(sys.argv[1:]) or "return window", top_k=8):
        flag = "PASS" if hit.score >= config.MIN_SCORE else "    "
        print(f"{hit.score:.3f} {flag} {hit.chunk.id}")
