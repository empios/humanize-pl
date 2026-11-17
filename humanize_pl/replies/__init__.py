"""Replying to an incoming letter: what it is, and which replies it allows.

`analyze(text)` reads any letter by its form (title, ruling words, sender,
cited codes, instruction), not by a list of authorities, so a letter from an
office it has never seen still gets its act and remedies. The knowledge lives
in `data/replies.yaml`. Given the delivery date, `analyze(text, delivered)`
also counts each term to its last day (`deadlines.py`).
"""

from humanize_pl.replies.analysis import Candidate, Guess, IncomingAnalysis, analyze
from humanize_pl.replies.catalogue import ReplyCatalogueError, Term, catalogue
from humanize_pl.replies.deadlines import Deadline, compute_deadline

__all__ = [
    "Candidate",
    "Deadline",
    "Guess",
    "IncomingAnalysis",
    "ReplyCatalogueError",
    "Term",
    "analyze",
    "catalogue",
    "compute_deadline",
]
