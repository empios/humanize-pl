from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from functools import lru_cache
import math
import os

import numpy as np


DEFAULT_SEMANTIC_MODEL = "sdadas/st-polish-paraphrase-from-distilroberta"
DEFAULT_FLUENCY_MODEL = "allegro/herbert-base-cased"
# Trained on NLI data in 26 languages, Polish among them; the former default
# (mDeBERTa-v3-base-mnli-xnli) saw none. On 318 Polish pairs, kept rewrites
# against changed modals, negations, numbers and swapped parties, it separates
# the two better (two-way entailment AUC 0.971 against 0.948) and its
# contradiction verdict catches more of every kind.
DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
# A rewrite must also entail its source and be entailed by it at least this
# much. A changed modality rarely reads as an outright contradiction: on 124
# kept rewrites and 194 changed ones, the contradiction verdict alone caught 17
# of 60 changed modals; with this floor 40, for 3 rejected good rewrites in
# 124 instead of 1.
MIN_MUTUAL_ENTAILMENT = 0.2

@dataclass(frozen=True)
class TransformerValidationResult:
    semantic_similarity: float | None = None
    fluency_delta: float | None = None
    semantic_model: str | None = None
    fluency_model: str | None = None
    nli_model: str | None = None


class NLIValidator:
    """NLI cross-encoder used to prevent hallucinations (contradictions)."""

    def __init__(self, model_name: str | None = None, *, offline: bool = False) -> None:
        self.model_name = model_name or DEFAULT_NLI_MODEL
        self.offline = offline
        self._pipeline = _get_nli_pipeline(self.model_name, offline)

    def check_entailment(self, premise: str, hypothesis: str) -> bool:
        """True if the hypothesis keeps the premise's meaning: the model does not
        call it a contradiction, and each entails the other at least
        MIN_MUTUAL_ENTAILMENT."""
        forward = self._scores(premise, hypothesis)
        if max(forward, key=forward.get) == "contradiction":
            return False
        backward = self._scores(hypothesis, premise)
        return min(forward["entailment"], backward["entailment"]) >= MIN_MUTUAL_ENTAILMENT

    def _scores(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Label -> probability; the labels are entailment, neutral, contradiction."""
        rows = self._pipeline({"text": premise, "text_pair": hypothesis}, top_k=None)
        return {row["label"].lower(): row["score"] for row in rows}


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


@lru_cache(maxsize=2)
def _get_nli_pipeline(model_name: str, offline: bool):
    from transformers import pipeline  # type: ignore

    with _offline_env(offline):
        return pipeline(
            "text-classification",
            model=model_name,
        )


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
