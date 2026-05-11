from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass
class TokenInfo:
    text: str
    lemma: str | None
    upos: str | None
    feats: str | None
    head: int | None
    deprel: str | None
    id: int | None = None
    start_char: int | None = None
    end_char: int | None = None


@dataclass
class SentenceAnalysis:
    tokens: list[TokenInfo]

    def lemmas(self) -> list[str]:
        return [token.lemma or token.text.lower() for token in self.tokens]

    def finite_verbs(self) -> list[TokenInfo]:
        return [token for token in self.tokens if _is_finite_verb(token)]

    def subjects(self) -> list[TokenInfo]:
        return [
            token
            for token in self.tokens
            if (token.deprel or "").lower() in {"nsubj", "nsubj:pass", "csubj"}
        ]

    def objects(self) -> list[TokenInfo]:
        return [
            token
            for token in self.tokens
            if (token.deprel or "").lower() in {"obj", "iobj"} or (token.deprel or "").lower().startswith("obl")
        ]

    def dependency_summary(self) -> dict[str, object]:
        finite = self.finite_verbs()
        subjects = self.subjects()
        objects = self.objects()
        subordinate_count = sum(
            1
            for token in self.tokens
            if (token.deprel or "").lower() in {"advcl", "acl", "ccomp", "xcomp", "csubj"}
        )
        passive_count = sum(
            1
            for token in self.tokens
            if (token.feats or "").lower().find("voice=pass") >= 0
            or (token.deprel or "").lower() == "nsubj:pass"
        )
        return {
            "lemmas": self.lemmas(),
            "finite_verbs": [token.text for token in finite],
            "subjects": [token.text for token in subjects],
            "objects": [token.text for token in objects],
            "subordinate_count": subordinate_count,
            "passive_count": passive_count,
            "has_finite_verb": bool(finite),
            "has_subject": bool(subjects),
            "has_object": bool(objects),
        }


class StanzaEngine:
    def __init__(self, *, offline: bool = False) -> None:
        self.offline = offline
        self._pipeline = _get_pipeline(offline)

    def analyze_sentence(self, sentence: str) -> SentenceAnalysis:
        doc = self._pipeline(sentence)
        tokens: list[TokenInfo] = []
        for sent in doc.sentences:
            for word in sent.words:
                parent_token = getattr(word, "parent", None)
                start_char = getattr(word, "start_char", None)
                end_char = getattr(word, "end_char", None)
                if start_char is None and parent_token is not None:
                    start_char = getattr(parent_token, "start_char", None)
                    end_char = getattr(parent_token, "end_char", None)
                tokens.append(
                    TokenInfo(
                        text=word.text,
                        lemma=getattr(word, "lemma", None),
                        upos=getattr(word, "upos", None),
                        feats=getattr(word, "feats", None),
                        head=getattr(word, "head", None),
                        deprel=getattr(word, "deprel", None),
                        id=getattr(word, "id", None),
                        start_char=start_char,
                        end_char=end_char,
                    )
                )
        return SentenceAnalysis(tokens=tokens)

    def finite_verbs(self, sentence: str) -> list[TokenInfo]:
        return self.analyze_sentence(sentence).finite_verbs()

    def subjects(self, sentence: str) -> list[TokenInfo]:
        return self.analyze_sentence(sentence).subjects()

    def objects(self, sentence: str) -> list[TokenInfo]:
        return self.analyze_sentence(sentence).objects()

    def lemmas(self, sentence: str) -> list[str]:
        return self.analyze_sentence(sentence).lemmas()

    def dependency_summary(self, sentence: str) -> dict[str, object]:
        return self.analyze_sentence(sentence).dependency_summary()


def _is_finite_verb(token: TokenInfo) -> bool:
    upos = (token.upos or "").upper()
    feats = token.feats or ""
    if upos not in {"VERB", "AUX"}:
        return False
    return "VerbForm=Fin" in feats or "Tense=" in feats or not feats


@lru_cache(maxsize=2)
def _get_pipeline(offline: bool):
    import stanza  # type: ignore
    from stanza.pipeline.core import DownloadMethod  # type: ignore

    return stanza.Pipeline(
        lang="pl",
        processors="tokenize,pos,lemma,depparse",
        tokenize_no_ssplit=True,
        download_method=DownloadMethod.NONE if offline else DownloadMethod.REUSE_RESOURCES,
        use_gpu=False,
        verbose=False,
    )
