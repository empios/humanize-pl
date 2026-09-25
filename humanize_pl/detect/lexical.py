"""Lexical diversity metrics that are robust to text length variations."""

from __future__ import annotations

import regex as re

WORD_RE = re.compile(r"\b\p{L}+\b")


def mtld(text: str, threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (MTLD).
    
    Evaluates the mean length of word strings that maintain a TTR above a 
    specified threshold (default 0.72). It calculates this forward and 
    backward, returning the average. This metric is significantly more robust 
    to text length variations than standard TTR or MSTTR.
    """
    tokens = [token.lower() for token in WORD_RE.findall(text)]
    if not tokens:
        return 0.0

    def _mtld_pass(token_list: list[str]) -> float:
        factors = 0.0
        types: set[str] = set()
        token_count = 0
        
        for token in token_list:
            token_count += 1
            types.add(token)
            ttr = len(types) / token_count
            
            if ttr <= threshold:
                factors += 1.0
                types.clear()
                token_count = 0
                
        # If the text doesn't perfectly divide, we add a partial factor
        if token_count > 0:
            ttr = len(types) / token_count
            # The partial factor formula for MTLD: (1 - ttr) / (1 - threshold)
            partial_factor = (1.0 - ttr) / (1.0 - threshold)
            factors += partial_factor
            
        return len(token_list) / factors if factors > 0 else 0.0

    forward = _mtld_pass(tokens)
    backward = _mtld_pass(tokens[::-1])
    
    return round((forward + backward) / 2.0, 4)


CONNECTIVES = [
    "ponadto", "jednakże", "tymczasem", "zatem", "więc", "mianowicie",
    "w konsekwencji", "co więcej", "jednak", "chociaż", "mimo to", "dlatego",
    "ponieważ", "z kolei", "reansumując", "reasumując", "konkludując",
    "nadto", "wobec tego", "co za tym idzie"
]

_CONNECTIVES_RE = re.compile(
    r"\b(" + "|".join(pattern.replace(" ", r"\s+") for pattern in CONNECTIVES) + r")\b",
    re.IGNORECASE
)


def connective_density(text: str) -> float:
    """Number of logical/discourse connectives per 1,000 words.
    
    AI models overuse transitional words (ponadto, jednakże, zatem) compared to
    human writers, making the text feel overly cohesive or 'slop'-like.
    """
    tokens = WORD_RE.findall(text)
    if not tokens:
        return 0.0
    
    word_count = len(tokens)
    connective_count = len(_CONNECTIVES_RE.findall(text))
    
    return round((connective_count / word_count) * 1000.0, 4)
