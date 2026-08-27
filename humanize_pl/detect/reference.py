from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from statistics import mean, pstdev

import regex as re


WORD_RE = re.compile(r"\p{L}+")

# Type-token ratio is length-dependent, so it is averaged over fixed windows
# instead of computed over whole documents of wildly differing size.
TTR_WINDOW = 200


@dataclass(frozen=True)
class Distribution:
    mean: float
    sd: float
    p50: float
    p90: float
    p95: float
    p99: float

    @classmethod
    def empty(cls) -> "Distribution":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def of(cls, values: list[float]) -> "Distribution":
        if not values:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ordered = sorted(values)
        return cls(
            mean=round(mean(ordered), 4),
            sd=round(pstdev(ordered), 4) if len(ordered) > 1 else 0.0,
            p50=round(_percentile(ordered, 0.50), 4),
            p90=round(_percentile(ordered, 0.90), 4),
            p95=round(_percentile(ordered, 0.95), 4),
            p99=round(_percentile(ordered, 0.99), 4),
        )


@dataclass(frozen=True)
class ReferenceProfile:
    """Human-writing baseline for a single genre of Polish legal text.

    Every number here is measured on human documents. The detection layer's
    thresholds mean nothing without it: a sentence-length CV of 0.52 is neither
    high nor low until there is something to compare it against.

    Genre matters. Court reasoning is not a client-facing opinion and not a
    contract; profiles are per genre on purpose and must not be mixed.
    """

    name: str
    genre: str
    source: str
    built_on: str
    document_count: int
    word_count: int
    sentence_count: int
    sentence_words: Distribution
    sentence_length_cv: Distribution
    paragraph_shape_cv: Distribution
    opening_diversity: Distribution
    windowed_ttr: Distribution
    anonymisation_rate: Distribution
    signal_score: Distribution
    family_rates: dict[str, Distribution] = field(default_factory=dict)

    def to_json(self) -> dict:
        payload = asdict(self)
        return payload

    @classmethod
    def from_json(cls, payload: dict) -> "ReferenceProfile":
        data = dict(payload)
        for key in (
            "sentence_words",
            "sentence_length_cv",
            "paragraph_shape_cv",
            "opening_diversity",
            "windowed_ttr",
            "anonymisation_rate",
            "signal_score",
        ):
            # A profile written before a metric existed loads with an empty
            # distribution rather than failing. Calibration skips zero-valued
            # baselines, so an older profile degrades instead of breaking —
            # and a new profile can be built while an older one is installed.
            data[key] = Distribution(**data[key]) if key in data else Distribution.empty()
        data["family_rates"] = {
            family: Distribution(**values)
            for family, values in (data.get("family_rates") or {}).items()
        }
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceProfile":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def windowed_ttr(text: str, *, window: int = TTR_WINDOW) -> float:
    tokens = [token.lower() for token in WORD_RE.findall(text)]
    if len(tokens) < window:
        return round(len(set(tokens)) / len(tokens), 4) if tokens else 0.0
    ratios = [
        len(set(tokens[start : start + window])) / window
        for start in range(0, len(tokens) - window + 1, window)
    ]
    return round(mean(ratios), 4)


def _percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
