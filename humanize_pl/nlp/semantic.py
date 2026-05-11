from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from functools import lru_cache
import math
import os

import numpy as np


DEFAULT_SEMANTIC_MODEL = "sdadas/st-polish-paraphrase-from-distilroberta"
DEFAULT_FLUENCY_MODEL = "allegro/herbert-base-cased"


@dataclass(frozen=True)
class TransformerValidationResult:
    semantic_similarity: float | None = None
    fluency_delta: float | None = None
    semantic_model: str | None = None
    fluency_model: str | None = None


class EmbeddingSimilarityValidator:
    def __init__(
        self,
        threshold: float = 0.94,
        model_name: str | None = None,
        *,
        offline: bool = False,
    ) -> None:
        self.threshold = threshold
        self.model_name = model_name or DEFAULT_SEMANTIC_MODEL
        self.offline = offline
        self._model = _get_embedding_model(self.model_name, offline)

    def similarity(self, a: str, b: str) -> float:
        emb = self._model.encode([a, b], normalize_embeddings=True)
        return float(np.dot(emb[0], emb[1]))

    def is_similar(self, a: str, b: str) -> bool:
        return self.similarity(a, b) >= self.threshold


class MaskedLMFluencyScorer:
    """Lightweight masked-LM sentence scorer used only as a validator/scorer."""

    def __init__(self, model_name: str | None = None, *, offline: bool = False) -> None:
        self.model_name = model_name or DEFAULT_FLUENCY_MODEL
        self.offline = offline
        self._tokenizer, self._model = _get_masked_lm(self.model_name, offline)

    def score(self, text: str) -> float:
        import torch  # type: ignore

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=160,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        with torch.no_grad():
            output = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
        loss = float(output.loss.detach().cpu())
        if math.isnan(loss) or math.isinf(loss):
            return -100.0
        return -loss

    def delta(self, original: str, candidate: str) -> float:
        return self.score(candidate) - self.score(original)


# Backward-compatible name used by earlier package versions.
SemanticValidator = EmbeddingSimilarityValidator


@lru_cache(maxsize=4)
def _get_embedding_model(model_name: str, offline: bool):
    from sentence_transformers import SentenceTransformer  # type: ignore

    with _offline_env(offline):
        return SentenceTransformer(model_name, local_files_only=offline)


@lru_cache(maxsize=4)
def _get_masked_lm(model_name: str, offline: bool):
    from transformers import AutoModelForMaskedLM, AutoTokenizer  # type: ignore

    with _offline_env(offline):
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=offline)
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            local_files_only=offline,
            use_safetensors=False if offline else None,
        )
    model.eval()
    return tokenizer, model


@contextmanager
def _offline_env(offline: bool):
    if not offline:
        yield
        return
    keys = (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "DISABLE_SAFETENSORS_CONVERSION",
        "HF_HUB_DISABLE_TELEMETRY",
    )
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
