"""Replying to an incoming letter: what it is, and which replies it allows.

`analyze(text)` reads any letter by its form (title, ruling words, sender,
cited codes, instruction), not by a list of authorities, so a letter from an
office it has never seen still gets its act and remedies. The knowledge lives
in `data/replies.yaml`.
"""

from humanize_pl.replies.analysis import Candidate, Guess, IncomingAnalysis, analyze
from humanize_pl.replies.catalogue import ReplyCatalogueError, Term, catalogue

__all__ = [
    "Candidate",
    "Guess",
    "IncomingAnalysis",
    "ReplyCatalogueError",
    "Term",
    "analyze",
    "catalogue",
]
