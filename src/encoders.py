"""Reprezentácie textu: viacjazyčný embeddingový model (hlavný baseline) a TF-IDF (kontrolný variant)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

MAX_CHARS = 3000  # dlhé abstrakty orezávame (model aj tak vidí max. 512 tokenov)


def _clip(texts: list[str]) -> list[str]:
    return [t[:MAX_CHARS] for t in texts]


class DenseEncoder:
    """Embeddingový model zo sentence-transformers, bez dolaďovania.

    Predvolený model intfloat/multilingual-e5-base vyžaduje prefixy:
    "query: " pre dokumenty (porovnávanie dokument-dokument je symetrické)
    a "passage: " pre heslá GND.
    """

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 64,
                 cache_dir: str | Path = "cache"):
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 512
        if device == "cuda":
            self.model.half()  # fp16: rýchlejšie a menej VRAM (RTX 3060 má 6 GB)
        self.is_e5 = "e5" in model_name.lower()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[encoder] {model_name} na zariadení: {device}")

    def _prefix(self, kind: str) -> str:
        if not self.is_e5:
            return ""
        return "query: " if kind == "doc" else "passage: "

    def encode(self, texts: list[str], kind: str = "doc", cache_key: str | None = None) -> np.ndarray:
        texts = [self._prefix(kind) + t for t in _clip(texts)]
        cache_file = None
        if cache_key:
            h = hashlib.md5(("\n".join(texts)).encode("utf-8")).hexdigest()[:10]
            safe_model = self.model_name.replace("/", "__")
            cache_file = self.cache_dir / f"{safe_model}__{cache_key}__{h}.npy"
            if cache_file.exists():
                print(f"[encoder] načítavam z cache: {cache_file.name}")
                return np.load(cache_file)
        emb = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=True)
        emb = emb.astype(np.float32)
        if cache_file is not None:
            np.save(cache_file, emb)
        return emb


class TfidfEncoder:
    """Klasický kontrolný variant bez neurónovej siete (slová + bigramy, sublineárne tf)."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300_000,
                                   sublinear_tf=True, strip_accents=None, lowercase=True)
        self.fitted = False

    def fit(self, texts: list[str]):
        self.vec.fit(_clip(texts))
        self.fitted = True
        return self

    def encode(self, texts: list[str], kind: str = "doc", cache_key: str | None = None):
        assert self.fitted, "TF-IDF treba najprv natrénovať na trénovacích textoch (fit)"
        return self.vec.transform(_clip(texts))  # riedka matica, riadky L2-normalizované
