"""Rule for breaking up long genitive chains (łańcuchy dopełniaczowe)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from humanize_pl.config import Mode
from humanize_pl.nlp.stanza_engine import SentenceAnalysis, TokenInfo

from .base import Candidate

logger = logging.getLogger(__name__)
_YAML = Path(__file__).parent / "genitive_chains.yaml"


@dataclass(frozen=True)
class ChainEntry:
    lemmas: tuple[str, ...]
    replacement: str
    risk: float
    modes: frozenset[str]


@lru_cache(maxsize=1)
def _load_chain_entries() -> list[ChainEntry]:
    with open(_YAML, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    
    entries = []
    for key, val in raw.items():
        lemmas = tuple(key.lower().split())
        entries.append(
            ChainEntry(
                lemmas=lemmas,
                replacement=val["replacement"],
                risk=float(val.get("risk", 0.5)),
                modes=frozenset(val.get("modes", ["standard", "strong"])),
            )
        )
    return entries


def _get_genitive_chains(analysis: SentenceAnalysis) -> list[list[TokenInfo]]:
    """Traverse the dependency tree to find chains of Genitive modifiers."""
    tokens = {tok.id: tok for tok in analysis.tokens}
    children = {tok.id: [] for tok in analysis.tokens}
    for tok in analysis.tokens:
        if tok.head and tok.head > 0:
            children[tok.head].append(tok)
            
    chains: list[list[TokenInfo]] = []
    
    # Only start chains from tokens that are NOT themselves Genitive modifiers
    # to avoid yielding overlapping sub-chains.
    for tok in analysis.tokens:
        if tok.head and tok.head > 0:
            head_tok = tokens.get(tok.head)
            # If this token is a genitive modifier of its head, it's not the start of a chain
            if tok.deprel in ("nmod", "nmod:arg", "obj", "iobj") and "Case=Gen" in (tok.feats or ""):
                continue
                
        # This token is a root of a potential chain
        chain = [tok]
        current = tok.id
        
        while True:
            # Find a Genitive dependent
            gens = [
                c for c in children.get(current, []) 
                if c.deprel in ("nmod", "nmod:arg", "obj", "iobj") 
                and "Case=Gen" in (c.feats or "")
                and c.upos in ("NOUN", "PROPN")
            ]
            if not gens:
                break
            
            # Follow the first genitive branch (usually linear in legal polish)
            chain.append(gens[0])
            current = gens[0].id
            
        if len(chain) >= 2:
            chains.append(chain)
            
    return chains


def genitive_chain_candidates(
    sentence: str, analysis: SentenceAnalysis | None, *, mode: Mode
) -> list[Candidate]:
    """Find and rewrite excessively long genitive chains."""
    if not analysis or not getattr(analysis, "tokens", None):
        return []

    active = mode.value
    entries = _load_chain_entries()
    chains = _get_genitive_chains(analysis)
    
    candidates: list[Candidate] = []
    
    for chain in chains:
        # Include prepositions attached to the root of the chain if present
        root_tok = chain[0]
        prep_tok = None
        for tok in analysis.tokens:
            if tok.head == root_tok.id and tok.deprel == "case":
                prep_tok = tok
                break
        
        # Build the full chain of tokens we might replace
        full_chain = [prep_tok] + chain if prep_tok else chain
        
        # The lemmas we want to match against the dictionary
        chain_lemmas = tuple((t.lemma or t.text).lower() for t in full_chain)
        
        for entry in entries:
            if active not in entry.modes:
                continue
                
            # Check if entry.lemmas is a prefix of our chain_lemmas
            if len(chain_lemmas) >= len(entry.lemmas) and chain_lemmas[:len(entry.lemmas)] == entry.lemmas:
                # We found a match!
                matched_tokens = full_chain[:len(entry.lemmas)]
                
                # Find the character span in the original sentence
                start_char = matched_tokens[0].start_char
                end_char = matched_tokens[-1].end_char
                
                if start_char is None or end_char is None:
                    continue
                    
                # Build the replacement
                replaced_text = sentence[:start_char] + entry.replacement + sentence[end_char:]
                
                # Match capitalization
                if sentence[start_char:start_char+1].isupper():
                    replaced_text = sentence[:start_char] + entry.replacement[:1].upper() + entry.replacement[1:] + sentence[end_char:]
                    
                candidates.append(
                    Candidate(
                        text=replaced_text,
                        rule="genitive_chains:simplify",
                        score=entry.risk,
                    )
                )
                
    return candidates
