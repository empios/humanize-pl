from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from functools import lru_cache
import math
import os
import threading

import numpy as np
import regex as re

from humanize_pl.sentence_splitter import split_sentences


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
# The model reads a pair of up to 512 tokens; past that the end of a long
# paragraph and its rewrite goes unread. On 20 long rewrites (518-713 tokens)
# a sentence invented at the end was caught in 4; split into aligned parts
# that each fit, in 19, for 17 good rewrites accepted instead of 19.
MAX_PAIR_TOKENS = 512
_WORD_RE = re.compile(r"\w{4,}")

@dataclass(frozen=True)
class TransformerValidationResult:
    semantic_similarity: float | None = None
    fluency_delta: float | None = None
    semantic_model: str | None = None
    fluency_model: str | None = None
    nli_model: str | None = None


# The hosted model's fragments are validated on several threads at once, and
# the NLI pipeline (cached, so shared) runs on the GPU: two concurrent calls
# crashed the process in Metal ("A command encoder is already encoding to
# this command buffer"). One inference at a time.
_NLI_LOCK = threading.Lock()


class NLIValidator:
    """NLI cross-encoder used to prevent hallucinations (contradictions)."""

    def __init__(self, model_name: str | None = None, *, offline: bool = False) -> None:
        self.model_name = model_name or DEFAULT_NLI_MODEL
        self.offline = offline
        self._pipeline = _get_nli_pipeline(self.model_name, offline)

    def check_entailment(self, premise: str, hypothesis: str) -> bool:
        """True if the hypothesis keeps the premise's meaning: the model does not
        call it a contradiction, and each entails the other at least
        MIN_MUTUAL_ENTAILMENT. A pair too long for the model is checked in
        aligned parts, every one of which must hold."""
        return all(
            self._mutual(part_premise, part_hypothesis)
            for part_premise, part_hypothesis in self._parts(premise, hypothesis)
        )

    def _mutual(self, premise: str, hypothesis: str) -> bool:
        forward = self._scores(premise, hypothesis)
        if max(forward, key=forward.get) == "contradiction":
            return False
        backward = self._scores(hypothesis, premise)
        return min(forward["entailment"], backward["entailment"]) >= MIN_MUTUAL_ENTAILMENT

    def _fits(self, premise: str, hypothesis: str) -> bool:
        tokenizer = getattr(self._pipeline, "tokenizer", None)
        if tokenizer is None:
            return True
        return len(tokenizer(premise, hypothesis)["input_ids"]) <= MAX_PAIR_TOKENS

    def _parts(self, premise: str, hypothesis: str) -> list[tuple[str, str]]:
        """The pair itself when it fits; otherwise the premise cut into the
        fewest runs of sentences that fit, and the hypothesis cut where its
        words best follow each run - a rewrite merges and splits sentences, so
        cutting both at the same sentence count misaligns them. The whole pair
        again when no cut fits."""
        if self._fits(premise, hypothesis):
            return [(premise, hypothesis)]
        source, rewrite = split_sentences(premise), split_sentences(hypothesis)
        for count in range(2, len(source) + 1):
            size = -(-len(source) // count)
            runs = [source[index : index + size] for index in range(0, len(source), size)]
            cuts = _aligned_cuts(runs, rewrite)
            if cuts is None:
                continue
            bounds = [0, *cuts, len(rewrite)]
            parts = [
                (" ".join(run), " ".join(rewrite[bounds[index] : bounds[index + 1]]))
                for index, run in enumerate(runs)
            ]
            if all(rewrite_part and self._fits(source_part, rewrite_part) for source_part, rewrite_part in parts):
                return parts
        return [(premise, hypothesis)]

    def _scores(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Label -> probability; the labels are entailment, neutral, contradiction."""
        with _NLI_LOCK:
            rows = self._pipeline({"text": premise, "text_pair": hypothesis}, top_k=None)
        return {row["label"].lower(): row["score"] for row in rows}


def _words(sentences: list[str]) -> set[str]:
    return {word.casefold() for sentence in sentences for word in _WORD_RE.findall(sentence)}


def _overlap(left: list[str], right: list[str]) -> float:
    a, b = _words(left), _words(right)
    return len(a & b) / max(1, len(a | b))


def _aligned_cuts(runs: list[list[str]], rewrite: list[str]) -> list[int] | None:
    """Where to cut the rewrite after each run but the last: at each boundary,
    the sentence position whose two sides share the most words with the source
    on either side of it. None when the rewrite has too few sentences."""
    cuts: list[int] = []
    start = 0
    for index in range(len(runs) - 1):
        left = [sentence for run in runs[: index + 1] for sentence in run]
        right = [sentence for run in runs[index + 1 :] for sentence in run]
        positions = range(start + 1, len(rewrite))
        if not positions:
            return None
        start = max(
            positions,
            key=lambda cut: _overlap(left, rewrite[:cut]) + _overlap(right, rewrite[cut:]),
        )
        cuts.append(start)
    return cuts


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
