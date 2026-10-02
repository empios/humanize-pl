"""Layout-faithful IR for Polish legal templates.

A template here is NOT a bag of section keywords. It is the document itself,
indexed: an ordered list of blocks (one per Word paragraph), each carrying

  * its structural role (title, clause_heading, item, blank_line, hint, ...),
  * its layout facts (alignment, indents, list marker as *displayed*, sizes,
    bold/italic spans, tabs, page breaks),
  * its text verbatim, with the fill-in blanks ("....", "…") promoted to named
    slots that remember their exact character span and the caption that
    explains them.

The DOCX itself stays the source of truth (nothing is lost); this IR is the
addressable index used to fill blanks and to compare other documents. Shipped
templates live in humanize_pl/data/templates/<category>/ (precise variants) and
inside humanize_pl/data/blueprints/<category>.yaml (the general document of a
category); the DOCX they came from is not shipped.

From the repo root:

    python -m humanize_pl.layout_ir outline wzor.docx
    python -m humanize_pl.layout_ir extract wzor.docx --category umowa_o_prace -o wzor.yaml
    python -m humanize_pl.layout_ir check wzor.yaml
"""

from __future__ import annotations

import argparse
import copy
import io
import re
import sys
import unicodedata
import zipfile
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import regex as _regex
import yaml
from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def q(tag: str) -> str:
    return f"{{{W}}}{tag}"


# ----------------------------------------------------------------- styles ---


def _onoff(el) -> bool:
    val = el.get(q("val"))
    return val not in ("0", "false", "off")


def parse_rpr(rpr) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if rpr is None:
        return out
    for child in rpr:
        tag = etree.QName(child).localname
        if tag == "sz":
            out["sz"] = int(child.get(q("val")))
        elif tag == "b":
            out["b"] = _onoff(child)
        elif tag == "i":
            out["i"] = _onoff(child)
        elif tag == "u":
            out["u"] = child.get(q("val")) not in ("none", None)
        elif tag == "vertAlign":
            out["vert"] = child.get(q("val"))
        elif tag == "caps":
            out["caps"] = _onoff(child)
    return out


def parse_ppr(ppr) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if ppr is None:
        return out
    for child in ppr:
        tag = etree.QName(child).localname
        if tag == "jc":
            out["jc"] = child.get(q("val"))
        elif tag == "ind":
            ind = {}
            for key, names in (
                ("left", ("left", "start")),
                ("right", ("right", "end")),
                ("first", ("firstLine",)),
                ("hanging", ("hanging",)),
            ):
                for name in names:
                    if child.get(q(name)) is not None:
                        ind[key] = int(child.get(q(name)))
            out["ind"] = ind
        elif tag == "numPr":
            num = child.find(q("numId"))
            lvl = child.find(q("ilvl"))
            out["num"] = {
                "numId": int(num.get(q("val"))) if num is not None else None,
                "ilvl": int(lvl.get(q("val"))) if lvl is not None else None,
            }
        elif tag == "pStyle":
            out["style"] = child.get(q("val"))
        elif tag == "keepNext":
            out["keepNext"] = _onoff(child)
        elif tag == "pageBreakBefore":
            out["pageBreakBefore"] = _onoff(child)
        elif tag == "tabs":
            out["tabs"] = len(child.findall(q("tab")))
    return out


def _merge(base: dict, top: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in top.items():
        if key == "ind" and "ind" in merged:
            merged["ind"] = {**merged["ind"], **value}
        elif key == "num" and "num" in merged:
            merged["num"] = {
                k: (value[k] if value.get(k) is not None else merged["num"].get(k)) for k in value
            }
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class Styles:
    def __init__(self, styles_xml: bytes | None):
        self.default_rpr: dict[str, Any] = {"sz": 22}
        self.default_ppr: dict[str, Any] = {}
        self.styles: dict[str, dict[str, Any]] = {}
        self.default_paragraph: str | None = None
        if not styles_xml:
            return
        root = etree.fromstring(styles_xml)
        defaults = root.find(q("docDefaults"))
        if defaults is not None:
            rpr = defaults.find(f"{q('rPrDefault')}/{q('rPr')}")
            self.default_rpr = {**self.default_rpr, **parse_rpr(rpr)}
            ppr = defaults.find(f"{q('pPrDefault')}/{q('pPr')}")
            self.default_ppr = parse_ppr(ppr)
        for style in root.findall(q("style")):
            sid = style.get(q("styleId"))
            based = style.find(q("basedOn"))
            name = style.find(q("name"))
            self.styles[sid] = {
                "type": style.get(q("type")),
                "name": name.get(q("val")) if name is not None else sid,
                "based": based.get(q("val")) if based is not None else None,
                "ppr": parse_ppr(style.find(q("pPr"))),
                "rpr": parse_rpr(style.find(q("rPr"))),
            }
            if style.get(q("type")) == "paragraph" and style.get(q("default")) == "1":
                self.default_paragraph = sid

    def chain(self, style_id: str | None) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        sid = style_id or self.default_paragraph
        while sid and sid in self.styles and sid not in seen:
            seen.add(sid)
            out.append(sid)
            sid = self.styles[sid]["based"]
        return list(reversed(out))

    def paragraph_props(self, style_id: str | None, direct: dict) -> tuple[dict, dict]:
        ppr = copy.deepcopy(self.default_ppr)
        rpr = copy.deepcopy(self.default_rpr)
        for sid in self.chain(style_id):
            ppr = _merge(ppr, self.styles[sid]["ppr"])
            rpr = {**rpr, **self.styles[sid]["rpr"]}
        ppr = _merge(ppr, direct)
        return ppr, rpr

    def style_name(self, style_id: str | None) -> str:
        if not style_id:
            return ""
        return self.styles.get(style_id, {}).get("name", style_id)


# -------------------------------------------------------------- numbering ---

_BULLETS = {"": "•", "": "▪", "": "➢", "": "✓", "": "❖", "": "•"}


def _fmt_counter(fmt: str, n: int) -> str:
    if fmt == "decimal":
        return str(n)
    if fmt == "decimalZero":
        return f"{n:02d}"
    if fmt in ("lowerLetter", "upperLetter"):
        s = ""
        m = n
        while m > 0:
            m, r = divmod(m - 1, 26)
            s = chr(ord("a") + r) + s
        return s.upper() if fmt == "upperLetter" else s
    if fmt in ("lowerRoman", "upperRoman"):
        vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
                (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
        s, m = "", n
        for v, sym in vals:
            while m >= v:
                s += sym
                m -= v
        return s.upper() if fmt == "upperRoman" else s
    return str(n)


class Numbering:
    def __init__(self, numbering_xml: bytes | None):
        self.abstract: dict[int, dict[int, dict]] = {}
        self.nums: dict[int, dict] = {}
        self.counters: dict[int, dict[int, int]] = {}
        self.seen_nums: set[int] = set()
        if not numbering_xml:
            return
        root = etree.fromstring(numbering_xml)
        for ab in root.findall(q("abstractNum")):
            aid = int(ab.get(q("abstractNumId")))
            levels = {}
            for lvl in ab.findall(q("lvl")):
                il = int(lvl.get(q("ilvl")))
                start = lvl.find(q("start"))
                fmt = lvl.find(q("numFmt"))
                text = lvl.find(q("lvlText"))
                levels[il] = {
                    "start": int(start.get(q("val"))) if start is not None else 1,
                    "fmt": fmt.get(q("val")) if fmt is not None else "decimal",
                    "text": text.get(q("val")) if text is not None else "%1.",
                    "ind": parse_ppr(lvl.find(q("pPr"))).get("ind", {}),
                }
            self.abstract[aid] = levels
        for num in root.findall(q("num")):
            nid = int(num.get(q("numId")))
            ab = num.find(q("abstractNumId"))
            overrides = {}
            for ov in num.findall(q("lvlOverride")):
                st = ov.find(q("startOverride"))
                if st is not None:
                    overrides[int(ov.get(q("ilvl")))] = int(st.get(q("val")))
            self.nums[nid] = {"abstract": int(ab.get(q("val"))), "overrides": overrides}

    def label(self, num_id: int, ilvl: int) -> tuple[str, str] | None:
        info = self.nums.get(num_id)
        if info is None:
            return None
        aid = info["abstract"]
        levels = self.abstract.get(aid, {})
        if ilvl not in levels:
            return None
        counters = self.counters.setdefault(aid, {})
        if num_id not in self.seen_nums:
            self.seen_nums.add(num_id)
            for lv, start in info["overrides"].items():
                counters[lv] = start - 1
        level = levels[ilvl]
        counters[ilvl] = counters.get(ilvl, level["start"] - 1) + 1
        for deeper in [k for k in counters if k > ilvl]:
            del counters[deeper]
        if level["fmt"] == "bullet":
            return _BULLETS.get(level["text"], level["text"] or "•"), "bullet"
        text = level["text"]

        def sub(match):
            lv = int(match.group(1)) - 1
            lv_fmt = levels.get(lv, level)["fmt"]
            lv_start = levels.get(lv, level)["start"]
            return _fmt_counter(lv_fmt, counters.get(lv, lv_start))

        return re.sub(r"%(\d)", sub, text), level["fmt"]


# --------------------------------------------------------------- patterns ---

# Fill-in blanks: runs of dots / ellipses (weighted: "…" counts as three dots)
# or underscores; a dotted postal code ("..-…") is one run. A lone "r." or
# "k.p." never reaches the threshold.
_FILL_RUN = re.compile(r"(?:…|\.)+(?:[-–](?:…|\.)+)?|_{4,}")
_FILL_MIN_WEIGHT = 4  # "…." and "...." are blanks: "w terminie …. dni"
_PLACEHOLDER = re.compile(
    r"\[[^\]\n]{0,60}(?:uzupełnij|do uzupełnienia|wpisz|tbd)[^\]\n]{0,60}\]", re.IGNORECASE
)
# Captions sit in round brackets or, in some templates, square ones:
# "(miejscowość)", "[dzień/miesiąc/rok]".
_PAREN_GROUPS = re.compile(r"\(([^()]+)\)|\[([^\[\]]+)\]")
_CONDITIONAL = re.compile(
    r"dotyczy|nie jest obowiązkow|w przypadku|jeżeli|jeśli|tylko |opcjonaln|rekomenduje|fakultatyw"
    r"|o ile |skreślić|niepotrzebne",
    re.IGNORECASE,
)
_INSTRUCTION = re.compile(
    r"należy|o której mowa|o którym mowa|art\.|dotyczy|nie jest obowiązkow|rekomenduje|np\.|"
    r"w przypadku|jeżeli|jeśli|zgodnie z",
    re.IGNORECASE,
)
_MANUAL_LABEL = _regex.compile(
    r"""^(?P<lead>\s*)
        (?P<label>
            §\s*\d+[a-z]?\s*[.)]?
          | \d+(?:\.\d+)*\s*[.)]
          | \(?[a-z]\)
          | [IVXLC]+[.)](?=\s+\p{Lu})
          | [-–—•·▪●○]
        )
        (?P<gap>\s+)(?=\S)""",
    _regex.X,
)
# "§ 3" alone in its paragraph: the clause title, if any, sits on the next line.
_BARE_PARAGRAPH = _regex.compile(r"^\s*(?P<label>§\s*\d+[a-z]?\s*[.)]?)\s*$")
# "zwanym dalej Najemcą", "zwanym w dalszej części umowy „Zleceniodawcą”",
# "zwanymi łącznie Stronami", "zwanym dalej Zamieniającym nr 1"
_ALIAS = re.compile(
    r"\bzwan\w*\s+(?:łącznie\s+)?"
    r"(?:(?:dalej(?:\s+jako)?|w\s+(?:dalszej\s+)?(?:treści|części)\s+\w+)\s+)?"
    r"(?:łącznie\s+)?[„\"]?(?P<alias>[A-ZĄĆĘŁŃÓŚŹŻ][\wąćęłńóśźż]+(?:\s+(?:nr\s+)?(?:\d+|[A-Z])\b)?)"
)
# The noun only: "(podpis Najemcy)" marks a signature, "data podpisania" a date.
_SIGNATURE_WORD = r"\bpodpis(?:u|y|ie|em|ów|ami|ach|om)?\b"
_SIGNATURE = re.compile(_SIGNATURE_WORD, re.IGNORECASE)
# One column of the line naming who signs: "Wynajmujący", "Za Dzierżawcę:", "Strona 1"
_PARTY_CELL = _regex.compile(r"(?:Za\s+)?\p{Lu}\w*(?:\s+(?:nr\s+)?\w+){0,2}\s*[:,]?")
_CLOSING = re.compile(r"z\s+(?:wyrazami\s+)?(?:poważaniem|szacunkiem)\s*[,.]?", re.IGNORECASE)
# What may part a caption from the blank it names: a little punctuation, or
# spaces and tabs on the same line ("………\t\t(podpis)").
_CAPTION_GAP = re.compile(r"[\s;,.]{0,4}|[ \t ]+")

_POLISH = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def slug(text: str, limit: int = 48) -> str:
    text = unicodedata.normalize("NFC", text).translate(_POLISH).casefold()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:limit].strip("_") or "pole"


def _weight(run: str) -> int:
    return sum(3 if c == "…" else 1 for c in run)


# ------------------------------------------------------------ data model ---


@dataclass
class Slot:
    id: str
    start: int
    end: int
    kind: str  # blank | placeholder
    blank: str
    type: str = "text"
    hint: str | None = None
    label: str | None = None
    context: str = ""


@dataclass
class Block:
    p: int  # paragraph index in document order (addressing for fill)
    role: str
    text: str
    align: str = "left"
    indent: dict = field(default_factory=dict)
    size: float | None = None
    bold: bool = False
    italic: bool = False
    spans: list = field(default_factory=list)
    label: str | None = None
    label_src: str | None = None  # auto | manual
    label_fmt: str | None = None
    level: int | None = None
    body_start: int = 0  # offset where text after a manual label begins
    style: str = ""
    in_table: bool = False
    tabs: int = 0
    page_break_before: bool = False
    slots: list = field(default_factory=list)
    inline_hints: list = field(default_factory=list)
    hint_kind: str | None = None  # label | instruction
    hint_for: int | None = None
    optional: bool = False
    condition: str | None = None
    alias: str | None = None
    alternatives: int = 0
    field_label: str | None = None


# -------------------------------------------------------------- extraction ---


def paragraph_segments(p) -> list[tuple[str, Any, str]]:
    """Ordered (kind, element, text) pieces of a paragraph: the single source of
    truth for character offsets, shared by extraction and filling."""
    out: list[tuple[str, Any, str]] = []
    for run in p.iter(q("r")):
        if next(run.iterancestors(q("p")), None) is not p:
            continue
        for child in run:
            tag = etree.QName(child).localname
            if tag == "t":
                out.append(("t", child, child.text or ""))
            elif tag == "tab":
                out.append(("tab", child, "\t"))
            elif tag == "br":
                kind = child.get(q("type"))
                out.append(("br", child, "\f" if kind == "page" else "\n"))
            elif tag == "noBreakHyphen":
                out.append(("hy", child, "-"))
    return out


def _run_props(run, styles: Styles, base_rpr: dict) -> dict:
    rpr = dict(base_rpr)
    rstyle = run.find(f"{q('rPr')}/{q('rStyle')}")
    if rstyle is not None:
        rpr = {**rpr, **styles.styles.get(rstyle.get(q("val")), {}).get("rpr", {})}
    rpr = {**rpr, **parse_rpr(run.find(q("rPr")))}
    return rpr


def _find_blanks(text: str) -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    weak: list[tuple[int, int]] = []  # too few dots alone: the ".." of "nr: .. .... ...."
    for m in _FILL_RUN.finditer(text):
        run = m.group(0)
        if run.startswith("_") or _weight(run) >= _FILL_MIN_WEIGHT:
            found.append((m.start(), m.end(), "blank"))
        elif _weight(run) >= 2 and (not m.start() or text[m.start() - 1].isspace()):
            weak.append((m.start(), m.end()))
    for m in _PLACEHOLDER.finditer(text):
        found.append((m.start(), m.end(), "placeholder"))
    found.sort()

    def wrapped(end: int, start: int) -> bool:  # parted only as a wrapped line parts them
        gap = text[end:start]
        return len(gap) <= 3 and not gap.strip(" \xa0") and "\t" not in gap

    # Blanks split only by spaces are one visual blank that wrapped.
    merged: list[tuple[int, int, str]] = []
    for item in found:
        if merged and item[2] == merged[-1][2] == "blank" and wrapped(merged[-1][1], item[0]):
            merged[-1] = (merged[-1][0], item[1], "blank")
        else:
            merged.append(item)
    for k, (start, end, kind) in enumerate(merged):
        lead = [a for a, z in weak if kind == "blank" and z <= start and wrapped(z, start)]
        if lead:
            merged[k] = (lead[-1], end, kind)
    # a list item that is nothing but "…" ("a) …" typed by hand, too)
    if not merged and len(weak) == 1:
        start, end = weak[0]
        if re.fullmatch(r"\s*(?:\(?[0-9a-z]{1,3}[.)])?\s*", text[:start] + text[end:]):
            merged = [(start, end, "blank")]
    return merged


def _manual_label(text: str) -> tuple[str, str, int, int] | None:
    """A list marker typed by hand: (label, format, level, offset of the text after it)."""
    m = _MANUAL_LABEL.match(text) or _BARE_PARAGRAPH.match(text)
    if not m:
        return None
    label = m.group("label").strip()
    if label.startswith("§"):
        return label, "paragraph", 0, m.end()
    if re.match(r"^\(?[a-z]\)$", label):
        return label, "lowerLetter", 1, m.end()
    if re.match(r"^[IVXLC]+[.)]$", label):
        return label, "upperRoman", 0, m.end()
    if re.match(r"^\d", label):
        return label, "decimal", 0 if "." not in label[:-1] else 1, m.end()
    return label, "bullet", 2, m.end()


def extract(source: bytes | str) -> dict[str, Any]:
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    z = zipfile.ZipFile(io.BytesIO(data))
    styles = Styles(z.read("word/styles.xml") if "word/styles.xml" in z.namelist() else None)
    numbering = Numbering(
        z.read("word/numbering.xml") if "word/numbering.xml" in z.namelist() else None
    )
    root = etree.fromstring(z.read("word/document.xml"))
    body = root.find(q("body"))

    blocks: list[Block] = []
    pending_break = False
    for index, p in enumerate(body.iter(q("p"))):
        direct = parse_ppr(p.find(q("pPr")))
        style_id = direct.get("style")
        ppr, base_rpr = styles.paragraph_props(style_id, direct)
        segments = paragraph_segments(p)
        text = "".join(t for _k, _e, t in segments)

        # run-level formatting -> dominant props + deviating spans
        pieces: list[tuple[int, int, dict]] = []
        cursor = 0
        for run in p.iter(q("r")):
            if next(run.iterancestors(q("p")), None) is not p:
                continue
            length = sum(
                len(
                    "\t"
                    if etree.QName(c).localname == "tab"
                    else "\n"
                    if etree.QName(c).localname == "br"
                    else "-"
                    if etree.QName(c).localname == "noBreakHyphen"
                    else (c.text or "")
                )
                for c in run
                if etree.QName(c).localname in ("t", "tab", "br", "noBreakHyphen")
            )
            if length:
                pieces.append((cursor, cursor + length, _run_props(run, styles, base_rpr)))
                cursor += length

        def sz_of(props, base=base_rpr):
            return (props.get("sz") or base.get("sz") or 22) / 2

        weight: dict[float, int] = {}
        for a, b, props in pieces:
            if text[a:b].strip():
                weight[sz_of(props)] = weight.get(sz_of(props), 0) + (b - a)
        size = max(weight, key=weight.get) if weight else None
        content = [(a, b, pr) for a, b, pr in pieces if text[a:b].strip()]
        bold = bool(content) and all(pr.get("b") for _a, _b, pr in content)
        italic = bool(content) and all(pr.get("i") for _a, _b, pr in content)
        spans = []
        for a, b, pr in content:
            row: dict[str, Any] = {}
            if size is not None and sz_of(pr) != size:
                row["size"] = sz_of(pr)
            if pr.get("b") and not bold:
                row["bold"] = True
            if pr.get("i") and not italic:
                row["italic"] = True
            if pr.get("vert") in ("superscript", "subscript"):
                row["vert"] = pr["vert"]
            if row:
                if spans and spans[-1]["end"] == a and {
                    k: v for k, v in spans[-1].items() if k not in ("start", "end")
                } == row:
                    spans[-1]["end"] = b
                else:
                    spans.append({"start": a, "end": b, **row})

        # list marker: auto (numbering.xml) or typed by hand
        label = label_src = label_fmt = None
        level = None
        body_start = 0
        num = ppr.get("num") or {}
        if num.get("numId") not in (None, 0):
            res = numbering.label(num["numId"], num.get("ilvl") or 0)
            if res:
                label, label_fmt = res
                label_src, level = "auto", num.get("ilvl") or 0
        if label is None:
            marker = _manual_label(text)
            if marker:
                label, label_fmt, level, body_start = marker
                label_src = "manual"

        indent = {k: v for k, v in (ppr.get("ind") or {}).items() if v}
        jc = {"both": "justify", "start": "left", "end": "right"}.get(ppr.get("jc"), ppr.get("jc"))
        block = Block(
            p=index,
            role="body",
            text=text,
            align=jc or "left",
            indent=indent,
            size=size,
            bold=bold,
            italic=italic,
            spans=spans,
            label=label,
            label_src=label_src,
            label_fmt=label_fmt,
            level=level,
            body_start=body_start,
            style=styles.style_name(style_id),
            in_table=next(p.iterancestors(q("tc")), None) is not None,
            tabs=text.count("\t"),
            page_break_before=pending_break or bool(ppr.get("pageBreakBefore")),
        )
        pending_break = False
        if "\f" in text and not text.replace("\f", "").strip():  # only a page break
            pending_break = True
            block.role = "pagebreak"
            blocks.append(block)
            continue
        if not text.strip():
            block.role = "spacer"
            blocks.append(block)
            continue

        for start, end, kind in _find_blanks(text):
            block.slots.append(Slot(id="", start=start, end=end, kind=kind, blank=text[start:end]))
        blocks.append(block)

    classify(blocks)
    name_slots(blocks)
    page = root.find(f".//{q('sectPr')}")
    page_info: dict[str, Any] = {}
    if page is not None:
        sz = page.find(q("pgSz"))
        mar = page.find(q("pgMar"))
        if sz is not None:
            page_info["size"] = [int(sz.get(q("w"), 0)), int(sz.get(q("h"), 0))]
        if mar is not None:
            page_info["margins"] = {
                k: int(mar.get(q(k), 0)) for k in ("top", "bottom", "left", "right")
            }
    return {
        "page": page_info,
        "default_size": styles.default_rpr.get("sz", 22) / 2,
        "blocks": [asdict(b) for b in blocks],
    }


def from_text(text: str) -> dict[str, Any]:
    """Plain text read the way `extract` reads a DOCX, one block per line.

    Markers, blanks and roles come out the same; layout does not, because a
    string carries none. Compare the result with `evaluate(..., layout=False)`.
    """
    blocks: list[Block] = []
    for index, line in enumerate(text.split("\n")):
        block = Block(p=index, role="body", text=line)
        if not line.strip():
            block.role = "spacer"
            blocks.append(block)
            continue
        marker = _manual_label(line)
        if marker:
            block.label, block.label_fmt, block.level, block.body_start = marker
            block.label_src = "manual"
        for start, end, kind in _find_blanks(line):
            block.slots.append(Slot(id="", start=start, end=end, kind=kind, blank=line[start:end]))
        blocks.append(block)
    classify(blocks)
    return {"page": {}, "default_size": None, "blocks": [asdict(b) for b in blocks]}


# ---------------------------------------------------------- classification ---

_CONNECTORS = {"a", "oraz", "i", "pomiędzy", "pomiedzy"}


def _captions(text: str) -> list[tuple[int, int, str]]:
    """(start, end, inner text) of each caption. A square bracket asking for input
    ("[wpisz …]") is a blank, not a caption."""
    return [
        (m.start(), m.end(), (m.group(1) or m.group(2)).strip())
        for m in _PAREN_GROUPS.finditer(text)
        if not _PLACEHOLDER.fullmatch(m.group(0))
    ]


def _hint_groups(text: str) -> list[str] | None:
    """`(a)   (b)` -> ['a', 'b'] when the whole paragraph is captions."""
    stripped = text.strip()
    if not stripped.startswith(("(", "[")):
        return None
    groups = _captions(stripped)
    rest = stripped
    for start, end, _inner in reversed(groups):
        rest = rest[:start] + rest[end:]
    if not groups or rest.strip(" \t\u00a0.,;"):
        return None
    return [inner for _start, _end, inner in groups]


def _is_only_blank(b: Block) -> bool:
    if not b.slots or any(s.kind != "blank" for s in b.slots):
        return False
    rest = "".join(
        c for k, c in enumerate(b.text) if not any(s.start <= k < s.end for s in b.slots)
    )
    return not re.sub(r"[\s,;.: ]", "", rest)


def _split_captions(content: list[Block]) -> dict[int, list[int]]:
    """Captions broken over paragraphs ("(dane Pracodawcy, … adres lub" /
    "określenie firmy, … NIP-u)"): the paragraphs of each, by paragraph."""
    split: dict[int, list[int]] = {}
    for pos, b in enumerate(content):
        if pos in split or not b.text.strip().startswith(("(", "[")):
            continue
        depth, parts = 0, []
        for k in range(pos, min(pos + 3, len(content))):
            text = content[k].text
            depth += text.count("(") + text.count("[") - text.count(")") - text.count("]")
            parts.append(k)
            if depth <= 0:
                break
        joined = " ".join(content[k].text.strip() for k in parts)
        if depth <= 0 and len(parts) > 1 and _hint_groups(joined):
            split.update(dict.fromkeys(parts, parts))
    return split


def classify(blocks: list[Block]) -> None:
    seen_title = False
    content = [b for b in blocks if b.role not in ("spacer", "pagebreak")]
    split = _split_captions(content)

    def caption(pos: int) -> str:  # a caption's text, joined over the paragraphs it spans
        parts = split.get(pos, [pos])
        if len(parts) == 1:
            return content[pos].text
        return " ".join(content[k].text.strip() for k in parts)

    for pos, b in enumerate(content):
        text = b.text
        body = text[b.body_start :].strip()
        groups = _hint_groups(caption(pos))
        # "(……)" with no words inside is a blank in brackets, not a hint
        if groups is not None and not re.search(r"[^\W\d_]", " ".join(groups)):
            groups = None

        if groups is not None:
            b.role = "hint"
            # dots inside a hint describe the field above ("od……… do………"); they are no field
            b.slots = []
            joined = " ".join(groups)
            b.hint_kind = (
                "instruction" if len(joined) > 70 or _INSTRUCTION.search(joined) else "label"
            )
            if _CONDITIONAL.search(joined):
                b.condition = joined
            continue

        if _is_only_blank(b) and b.label is None:
            b.role = "signature_line" if len(b.slots) >= 2 and "\t" in text else "blank_line"
            continue

        # "§ 3" alone on its line heads a clause whatever the length of its title
        if b.label_fmt == "paragraph" and (len(body) <= 60 or "\n" in text[: b.body_start]):
            b.role = "clause_heading"
            continue

        if (
            not seen_title
            and pos < 8
            and b.align == "center"
            and (b.bold or (b.size or 0) >= 13)
            and 2 <= len(body) <= 80
        ):
            b.role = "title"
            seen_title = True
            continue

        lab = re.match(r"^([^:…._]{2,70}?)\s*:\s*(.*)$", body, re.DOTALL)
        if lab:
            b.field_label = lab.group(1).strip()

        if b.label is not None:
            b.role = "item"
            continue

        m = _ALIAS.search(text)
        if m:
            b.role = "party_alias"
            b.alias = m.group("alias")
            continue

        if body.casefold().strip(" .,:") in _CONNECTORS and len(body) <= 10:
            b.role = "party_connector"
            continue
        if (
            lab
            and b.slots
            and not lab.group(2).strip(" ").startswith("(")
            and b.slots[0].start >= text.index(":")
        ):
            b.role = "field"
            continue
        if body.endswith(":") and len(body) <= 70 and not b.slots:
            b.role = "field_header"
            continue
        b.role = "body"

    # inline hints: "(...)" right after a blank inside the same paragraph
    for b in blocks:
        if b.role in ("hint", "spacer", "pagebreak"):
            continue
        for start, end, inner in _captions(b.text):
            before = b.text[:start]
            # "(słownie: ……)" holds a field of its own; dots describe the blank
            # only in an example or a note ("np. umowy z dnia ……")
            plain = not _find_blanks(inner) or _INSTRUCTION.search(inner)
            trails = bool(before) and re.search(r"(?:…|\.{3,}|_{3,})[\s;,.]{0,4}$", before)
            trails = trails or re.search(r"(?:…|\.{3,}|_{3,})[ \t ]+$", before)
            # "w [miejscowość] ………": a square-bracket caption may also lead its blank
            leads = b.text[start] == "[" and re.match(r"[\s;,.]{0,4}(?:…|\.{3,}|_{3,})", b.text[end:])
            if plain and (trails or leads):
                b.inline_hints.append({"start": start, "end": end, "text": inner})
                if _CONDITIONAL.search(inner):
                    b.optional, b.condition = True, inner

    # bind hint blocks to the block they explain
    last: Block | None = None
    for pos, b in enumerate(content):
        if b.role == "hint":
            parts = split.get(pos, [pos])
            if last is not None and pos != parts[0]:
                b.hint_for = last.p  # the rest of a caption, bound with its first line
            elif last is not None and not (last.slots and all(s.hint for s in last.slots)):
                # dots that have their caption already leave this one to what follows
                b.hint_for = last.p
                groups = _hint_groups(caption(pos)) or []
                if last.role == "clause_heading":
                    if b.condition:
                        last.optional, last.condition = True, b.condition
                else:
                    for k, slot in enumerate(last.slots):
                        if k < len(groups):
                            slot.hint = groups[k]
                        elif len(groups) == 1:
                            slot.hint = groups[0]
                    if b.condition:
                        last.optional, last.condition = True, b.condition
            continue
        last = b

    # alternatives: "a / b / c" where several parts carry blanks
    for b in blocks:
        if b.role in ("hint", "spacer", "pagebreak"):
            continue
        b.alternatives = _alternatives(b.text)


def _alternatives(text: str) -> int:
    """Parts of an "a / b / c" choice, 0 when the text offers none. A slash
    inside a caption ("(dzień/miesiąc/rok)") describes one field, not a choice."""
    parts = re.split(r"\s*/\s*", _PAREN_GROUPS.sub(" ", text))
    if len(parts) >= 3 and sum(1 for part in parts if _find_blanks(part)) >= 2:
        return len(parts)
    return 0


_TYPE_RULES = [
    ("place_date", r"miejscowo\w*\s+i\s+data|data\s+i\s+miejscowo"),
    ("pesel", r"pesel"),
    ("id_number", r"\bnip\b|\bkrs\b|regon|ceidg|dowod|dokument"),
    ("signature", _SIGNATURE_WORD),
    ("amount", r"wynagrodzen|słownie|slownie|\bzł\b|kwot|wysokości|wysokosci"),
    ("date", r"\bdata\b|\bdniu\b|\bdnia\b|\bdzień\b|\bod\b|\bdo\b|\br\.$"),
    ("address", r"adres|siedzib|zamieszkania|\bul\."),
    ("person", r"imię|imie|nazwisk|reprezentowan|przez"),
    ("company", r"nazwa firmy|firm"),
    ("place", r"miejscowo|miejsce|\bw$"),
]


def _guess_type(*hints: str | None) -> str:
    blob = " ".join(h for h in hints if h).casefold()
    for name, pattern in _TYPE_RULES:
        if re.search(pattern, blob):
            return name
    return "text"


def caption_follows(caption: dict, blank: tuple[int, int], text: str) -> bool:
    """Whether a caption names the blank just before it: "……… (miejscowość)",
    "………\t\t(podpis)"."""
    return caption["start"] >= blank[1] and bool(
        _CAPTION_GAP.fullmatch(text[blank[1] : caption["start"]])
    )


def caption_leads(
    caption: dict, blank: tuple[int, int], blanks: list[tuple[int, int]], text: str
) -> bool:
    """Whether a caption names the blank right after it ("[miejscowość] ………"),
    which it does only when it does not already follow another blank."""
    follows = any(caption_follows(caption, other, text) for other in blanks)
    return not follows and caption["end"] <= blank[0] <= caption["end"] + 5


def name_slots(blocks: list[Block]) -> None:
    """Readable id from the caption, else from the field label (own or inherited
    from the header line above), else positional. The wording around the blank
    is kept as `context` so a UI can show it; the id is never guessed from it."""
    used: dict[str, int] = {}
    header: str | None = None  # label of the closest header line above
    for b in blocks:
        if b.role in ("spacer", "pagebreak"):
            continue
        if b.role == "hint":
            continue
        own_header = None
        if b.role in ("item", "field_header", "body") and not b.slots:
            body = b.text[b.body_start :].strip()
            m = re.match(r"^([^:…._]{2,70}?)\s*:\s*$", body)
            if m:
                own_header = m.group(1).strip()
        captions = [(m.start(), m.end()) for m in _PAREN_GROUPS.finditer(b.text)]
        for k, s in enumerate(b.slots):
            # the wording around the blank, without the other blanks and the
            # captions it is not part of ("(słownie: ……)" keeps "słownie")
            text = b.text
            hidden = [(a, z) for a, z in captions if not a <= s.start < z]
            for start, end in hidden + [(o.start, o.end) for o in b.slots if o is not s]:
                text = text[:start] + " " * (end - start) + text[end:]
            before = text[: s.start]
            after = text[s.end :]
            def _words(chunk: str) -> list[str]:
                return [w for w in re.findall(r"[\wąćęłńóśźż§.…]+", chunk) if w.strip(".…")]

            words_before = _words(before[b.body_start :])[-4:]
            words_after = _words(after)[:2]
            s.context = " ".join(words_before) + " ▢ " + " ".join(words_after)
            label = None
            if b.field_label and s.start >= b.text.find(":") >= 0:
                label = b.field_label
            elif b.role in ("blank_line", "party_alias", "field") and header:
                label = header
            s.label = label
            hint = s.hint
            for ih in b.inline_hints:
                if hint is None and caption_follows(ih, (s.start, s.end), b.text):
                    hint = s.hint = ih["text"]
            blanks = [(o.start, o.end) for o in b.slots]
            for ih in b.inline_hints:
                if hint is None and caption_leads(ih, (s.start, s.end), blanks, b.text):
                    hint = s.hint = ih["text"]
            hint_is_instruction = bool(hint) and (
                len(hint) > 40 or bool(_INSTRUCTION.search(hint))
            )
            base = None
            if hint and not hint_is_instruction:
                base = hint
            elif label:
                base = label
            elif hint:
                base = hint
            if base is None:
                base = f"p{b.p}_{k + 1}"
            ident = slug(base)
            used[ident] = used.get(ident, 0) + 1
            if used[ident] > 1:
                ident = f"{ident}_{used[ident]}"
            s.id = ident
            # its own lead-in only: "w dniu …… (data) ……" says nothing of the second blank
            lead = max([o.end for o in b.slots if o.end <= s.start] + [b.body_start])
            own = _words(text[lead : s.start])
            s.type = _guess_type(
                hint if not hint_is_instruction else None, label, own[-1] if own else ""
            )
            previous = next((o for o in b.slots[:k] if o.end == lead), None)
            joined = not re.search(r"[(\[]", b.text[lead : s.start])
            if s.type == "text" and not own and previous and joined:
                s.type = previous.type  # "……/……/…… r.": parts of one value
            # "(własnoręczny podpis osoby udzielającej pełnomocnictwa)" is long, still a signature
            if hint and _SIGNATURE.search(hint):
                s.type = "signature"
        if own_header:
            header = own_header
        elif b.role not in ("blank_line", "party_alias", "field"):
            header = None
    _sign_here(blocks)


def _party_cells(line: str) -> list[str]:
    """'Wynajmujący\t\tNajemca' -> ['Wynajmujący', 'Najemca']: the columns of a
    line naming who signs. Empty for any other line."""
    cells = [cell.strip() for cell in re.split(r"\t+|\s{2,}", line) if cell.strip()]
    return cells if all(_PARTY_CELL.fullmatch(cell) for cell in cells) else []


def _defined(name: str, text: str) -> bool:
    """Whether a capitalised name is a term the document uses mid-sentence, as
    it does a party ("… a Najemcą", "… przez Zamawiającego")."""
    word = re.sub(r"^Za\s+", "", name).split()[0].rstrip(":,")
    stem = word[:1].upper() + word[1:].lower()
    stem = stem[: max(4, len(stem) - 2)]
    return bool(re.search(rf"[\w,;:)][ \t]+{re.escape(stem)}", text))


def _sign_here(blocks: list[Block]) -> None:
    """Dots with no caption, by the names of the signing parties, are where
    they sign: "……… ⇥ ………" over "Wynajmujący ⇥ Najemca", dots over
    "Zamawiający" or under "Z poważaniem". Nothing there is data to fill.

    Only after the last clause heading: before it, dots beside a party's name
    take the party's details.
    """
    lines: list[tuple[Block, int, int]] = []
    for b in blocks:
        if b.role in ("spacer", "pagebreak"):
            continue
        start = 0
        for part in b.text.split("\n"):
            lines.append((b, start, start + len(part)))
            start += len(part) + 1

    def slots_on(k: int) -> list[Slot]:
        b, start, end = lines[k]
        return [s for s in b.slots if start <= s.start and s.end <= end]

    def words(k: int) -> str:
        b, start, end = lines[k]
        text = b.text[start:end]
        for s in sorted(slots_on(k), key=lambda s: -s.start):
            text = text[: s.start - start] + " " * (s.end - s.start) + text[s.end - start :]
        return text.strip()

    def dotted(k: int) -> bool:
        return bool(slots_on(k)) and not re.search(r"\w", words(k))

    def says_sign(k: int) -> bool:  # "(podpis Najemcy)", "podpis Wykonawcy", "Podpisy wspólników:"
        text = words(k)
        short = lines[k][0].role == "hint" or text.startswith(("(", "[")) or len(text.split()) <= 6
        return short and bool(_SIGNATURE.search(text))

    def signing(k: int) -> bool:  # a line that belongs where the parties sign
        text = words(k)
        return (
            not text
            or dotted(k)
            or lines[k][0].role == "hint"
            or says_sign(k)
            or bool(_party_cells(text) or _CLOSING.fullmatch(text))
        )

    document = "\n".join(b.text for b in blocks)
    headings = [k for k, (b, _start, _end) in enumerate(lines) if b.role == "clause_heading"]
    for k in range(headings[-1] + 1 if headings else 0, len(lines)):
        free = [s for s in slots_on(k) if s.hint is None and s.type != "signature"]
        if not free or not dotted(k):
            continue
        around = []
        for step in (-1, 1):
            j = k + step
            while 0 <= j < len(lines) and (not words(j) or dotted(j)):
                j += step
            around.append(j if 0 <= j < len(lines) else None)
        above, below = around
        signs = False
        for j, side in ((above, "above"), (below, "below")):
            if j is None:
                continue
            text, cells = words(j), _party_cells(words(j))
            single = len(cells) == 1 and _defined(cells[0], document)
            signs = signs or bool(
                says_sign(j)
                or len(cells) >= 2
                or (side == "above" and _CLOSING.fullmatch(text))
                or (side == "below" and single and not text.endswith(":"))
                # "Zleceniodawca:" over the last dots of the document
                or (side == "above" and single and all(map(signing, range(k + 1, len(lines)))))
            )
        if signs:
            for s in free:
                s.type = "signature"


# -------------------------------------------------------------- rendering ---


def marked(ir: dict[str, Any], width: int = 100) -> str:
    """Human-readable annotated outline: role | layout | text with {{slots}}."""
    lines = []
    collapsed = 0
    for b in ir["blocks"]:
        if b["role"] == "spacer":
            collapsed += 1
            continue
        if collapsed:
            lines.append(f"       · ·  (+{collapsed} pusty akapit)")
            collapsed = 0
        text = b["text"]
        for s in sorted(b["slots"], key=lambda s: -s["start"]):
            text = text[: s["start"]] + "{{" + s["id"] + "}}" + text[s["end"] :]
        text = text.replace("\t", "⇥").replace("\f", "⏎strona").replace("\n", "↵")
        text = re.sub(r"\s{3,}", "  ", text)
        bits = [b["align"]]
        ind = {k: v for k, v in b["indent"].items() if not (k == "right" and v < 300)}
        if ind:
            bits.append("ind=" + ",".join(f"{k[0]}{v}" for k, v in ind.items()))
        if b["size"] and b["size"] != ir["default_size"]:
            bits.append(f"{b['size']:g}pt")
        if b["bold"]:
            bits.append("B")
        if b["italic"]:
            bits.append("I")
        if b["label"]:
            bits.append(f"[{b['label']}]" + ("*" if b["label_src"] == "auto" else ""))
        if b["optional"]:
            bits.append("OPCJONALNE")
        if b["alternatives"]:
            bits.append(f"wybór×{b['alternatives']}")
        if b["page_break_before"]:
            bits.append("⏎strona")
        extra = ""
        if b["role"] == "hint":
            extra = f"  → #{b['hint_for']}" + (f" [{b['hint_kind']}]" if b["hint_kind"] else "")
        lines.append(f"{b['p']:03d} {b['role']:<15}{' '.join(bits):<42}{text[:width]}{extra}")
    return "\n".join(lines)


# ------------------------------------------------------------------ YAML ---


_POSITIONAL_ID = re.compile(r"p\d+_\d+")


class _Dumper(yaml.SafeDumper):
    """SafeDumper that can print small mappings and lists inline."""


class _Flow(dict):
    pass


class _FlowList(list):
    pass


_Dumper.add_representer(
    _Flow, lambda d, v: d.represent_mapping("tag:yaml.org,2002:map", v, flow_style=True)
)
_Dumper.add_representer(
    _FlowList, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True)
)


def _layout(b: dict[str, Any]) -> _Flow:
    row: dict[str, Any] = {}
    if b["align"] != "left":
        row["align"] = b["align"]
    if b["indent"]:
        row["indent"] = _Flow(b["indent"])
    if b["size"]:
        row["size"] = b["size"]
    if b["bold"]:
        row["bold"] = True
    if b["italic"]:
        row["italic"] = True
    if b["label"]:
        row["marker"] = _Flow(
            text=b["label"], kind=b["label_fmt"], src=b["label_src"], level=b["level"]
        )
    if b["spans"]:
        row["spans"] = _FlowList(_Flow(s) for s in b["spans"])
    if b["tabs"]:
        row["tabs"] = b["tabs"]
    if b["page_break_before"]:
        row["page_break_before"] = True
    return _Flow(row)


def _slot_row(s: dict[str, Any]) -> _Flow:
    row = _Flow(id=s["id"], span=_FlowList([s["start"], s["end"]]), type=s["type"])
    optional = {
        "hint": s["hint"],
        "label": s["label"] if s["label"] != s["hint"] else None,
        # the wording around a blank is only worth keeping while its id is a placeholder
        "context": s["context"].strip() if _POSITIONAL_ID.fullmatch(s["id"]) else None,
    }
    row.update({k: v for k, v in optional.items() if v})
    return row


def to_yaml(
    ir: dict[str, Any], *, template: str | None, category: str, label_pl: str | None = None
) -> str:
    """The template file: one paragraph per block, layout and fields inline.

    `template` names the source DOCX; a document written directly as YAML has none.
    """
    out: dict[str, Any] = {}
    if template:
        out["template"] = template
    out["category"] = category
    if label_pl:
        out["label_pl"] = label_pl
    out |= {
        "default_size": ir["default_size"],
        "page": _Flow(
            size=_FlowList(ir["page"].get("size", [])),
            margins=_Flow(ir["page"].get("margins", {})),
        ),
    }
    blocks: list[Any] = []
    for b in ir["blocks"]:
        row: dict[str, Any] = {"p": b["p"], "role": b["role"]}
        if b["role"] == "spacer":
            if b["page_break_before"]:
                row["page_break_before"] = True
            blocks.append(_Flow(row))
            continue
        row["layout"] = _layout(b)
        row["text"] = b["text"]
        if b["slots"]:
            row["slots"] = [_slot_row(s) for s in b["slots"]]
        if b["inline_hints"]:
            row["inline_hints"] = [
                _Flow(span=_FlowList([h["start"], h["end"]]), text=h["text"])
                for h in b["inline_hints"]
            ]
        if b["hint_for"] is not None:
            row["hint_for"] = b["hint_for"]
            row["hint_kind"] = b["hint_kind"]
        for key in ("alias", "alternatives"):
            if b[key]:
                row[key] = b[key]
        if b["optional"]:
            row["optional"] = True
            row["condition"] = b["condition"]
        blocks.append(row)
    out["blocks"] = blocks
    return yaml.dump(out, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=140)


def from_yaml(text: str) -> dict[str, Any]:
    """Read a template file back into the structure `fill` and `evaluate` use."""
    return from_data(yaml.safe_load(text))


def from_data(data: dict[str, Any]) -> dict[str, Any]:
    """`from_yaml` for a mapping already parsed, e.g. a document nested in a blueprint."""
    blocks = []
    for row in data["blocks"]:
        layout = row.get("layout", {})
        marker = layout.get("marker") or {}
        body = row.get("text", "")
        body_start = 0
        if marker.get("src") == "manual":
            found = _manual_label(body)
            body_start = found[3] if found else 0
        slots = []
        for s in row.get("slots", []):
            start, end = s["span"]
            slots.append(
                asdict(
                    Slot(
                        id=s["id"],
                        start=start,
                        end=end,
                        kind="placeholder" if body[start : start + 1] == "[" else "blank",
                        blank=body[start:end],
                        type=s.get("type", "text"),
                        hint=s.get("hint"),
                        label=s.get("label"),
                        context=s.get("context", ""),
                    )
                )
            )
        block = Block(
            p=row["p"],
            role=row["role"],
            text=body,
            align=layout.get("align", "left"),
            indent=layout.get("indent", {}),
            size=layout.get("size"),
            bold=layout.get("bold", False),
            italic=layout.get("italic", False),
            spans=layout.get("spans", []),
            label=marker.get("text"),
            label_src=marker.get("src"),
            label_fmt=marker.get("kind"),
            level=marker.get("level"),
            body_start=body_start,
            tabs=layout.get("tabs", 0),
            page_break_before=layout.get("page_break_before", row.get("page_break_before", False)),
            slots=slots,
            inline_hints=[
                {"start": h["span"][0], "end": h["span"][1], "text": h["text"]}
                for h in row.get("inline_hints", [])
            ],
            hint_kind=row.get("hint_kind"),
            hint_for=row.get("hint_for"),
            optional=row.get("optional", False),
            condition=row.get("condition"),
            alias=row.get("alias"),
            alternatives=row.get("alternatives", 0),
        )
        blocks.append(asdict(block))
    return {
        "template": data.get("template"),
        "category": data.get("category"),
        "label_pl": data.get("label_pl"),
        "default_size": data.get("default_size", 11),
        "page": data.get("page", {}),
        "blocks": blocks,
    }


# ------------------------------------------------------------------- fill ---


def _patch(p, start: int, end: int, value: str) -> None:
    segments = paragraph_segments(p)
    cursor = 0
    first = True
    for kind, element, text in segments:
        a, b = cursor, cursor + len(text)
        cursor = b
        if b <= start or a >= end:
            continue
        if kind != "t":
            raise ValueError("a slot may not span a tab or a line break")
        lo, hi = max(start, a) - a, min(end, b) - a
        element.text = text[:lo] + (value if first else "") + text[hi:]
        element.set(XML_SPACE, "preserve")
        first = False


def fill(source: bytes | str, ir: dict[str, Any], values: dict[str, str]) -> bytes:
    """Write values into the blanks of the ORIGINAL docx: only text nodes inside
    a slot change; every run, indent, tab, style and page break stays."""
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    zin = zipfile.ZipFile(io.BytesIO(data))
    root = etree.fromstring(zin.read("word/document.xml"))
    paragraphs = list(root.find(q("body")).iter(q("p")))
    for block in ir["blocks"]:
        todo = [(s["start"], s["end"], values[s["id"]]) for s in block["slots"] if s["id"] in values]
        if not todo:
            continue
        p = paragraphs[block["p"]]
        if "".join(t for _k, _e, t in paragraph_segments(p)) != block["text"]:
            raise ValueError(f"akapit {block['p']} nie zgadza się z blueprintem")
        for start, end, value in sorted(todo, reverse=True):
            _patch(p, start, end, value)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            payload = zin.read(item.filename)
            if item.filename == "word/document.xml":
                payload = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            zout.writestr(item, payload)
    return out.getvalue()


# --------------------------------------------------------------- evaluate ---

_TOKEN = re.compile(r"[\wąćęłńóśźż§]+", re.UNICODE)
_ANCHOR_ROLES = {
    "title", "body", "item", "clause_heading", "field", "field_header", "party_alias",
    "party_connector",
}
_DEPENDENT = ("hint", "blank_line", "signature_line")
_FILL_ONLY = re.compile(r"^[\s….,;:_ -]*$")
MATCH = 0.6  # a block "is" the blueprint block
CHANGED = 0.35  # below MATCH but clearly the same place, reworded


def static_text(b: dict[str, Any]) -> str:
    spans = sorted(
        [(s["start"], s["end"]) for s in b["slots"]]
        + [(h["start"], h["end"]) for h in b["inline_hints"]]
    )
    out, last = [], 0
    for a, z in spans:
        out.append(b["text"][last:a])
        last = z
    out.append(b["text"][last:])
    return "".join(out)[b["body_start"] :]


def tokens(text: str) -> list[str]:
    return [t.casefold() for t in _TOKEN.findall(text)]


def text_lines(ir: dict[str, Any]) -> dict[str, Any]:
    """The template as plain text shows it: one block per line.

    Text cannot tell a soft line break from a new paragraph, so a block that
    breaks its line would face two candidate blocks and match neither. A blank
    or caption stays with the line it sits on; a caption across a break goes.
    """
    blocks: list[dict[str, Any]] = []
    for b in ir["blocks"]:
        if "\n" not in b["text"]:
            blocks.append(b)
            continue
        offset = 0
        for index, line in enumerate(b["text"].split("\n")):
            end = offset + len(line)
            blocks.append(
                {
                    **b,
                    "text": line,
                    "body_start": b["body_start"] if index == 0 else 0,
                    "label": b["label"] if index == 0 else None,
                    "spans": [],
                    "slots": [
                        {**s, "start": s["start"] - offset, "end": s["end"] - offset}
                        for s in b["slots"]
                        if offset <= s["start"] and s["end"] <= end
                    ],
                    "inline_hints": [
                        {**h, "start": h["start"] - offset, "end": h["end"] - offset}
                        for h in b["inline_hints"]
                        if offset <= h["start"] and h["end"] <= end
                    ],
                    "alternatives": _alternatives(line),
                }
            )
            offset = end + 1
    return {**ir, "blocks": blocks}


def fixed_words(ir: dict[str, Any]) -> list[str]:
    """The words a template fixes: its anchors' text, blanks and captions left out."""
    return [
        word for b in ir["blocks"] if b["role"] in _ANCHOR_ROLES for word in tokens(static_text(b))
    ]


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _expected(b: dict[str, Any]) -> tuple[list[str], int]:
    """Static tokens of a blueprint block and how many of them a candidate must
    carry. Alternatives ("a / b / c") are one-of: a candidate that picked one
    is complete, so only the longest alternative counts."""
    text = static_text(b)
    toks = tokens(text)
    need = len(toks)
    if b["alternatives"]:
        segs = [len(tokens(part)) for part in re.split(r"\s*/\s*", text)]
        need = len(toks) - (sum(segs) - max(segs))
    return toks, max(need, 1)


def _sim(b: dict[str, Any], c: dict[str, Any], *, strict: bool = False) -> float:
    """How well candidate block c carries blueprint block b's static words."""
    bt, need = _expected(b)
    if not bt:
        return 0.0
    if (b["slots"] or b["inline_hints"]) and not b["alternatives"] and len(bt) <= 6:
        pattern, _names = _template_regex(b)
        try:
            if re.match(r"\s*" + pattern, c["text"][c["body_start"] :], re.DOTALL):
                return 1.0
        except re.error:
            pass
    ct = tokens(c["text"][c["body_start"] :]) if not strict else tokens(static_text(c))
    common = _lcs(bt, ct)
    recall = min(1.0, common / need)
    ends_in_slot = bool(b["slots"]) and b["slots"][-1]["end"] >= len(b["text"].rstrip()) - 2
    if len(bt) <= 3:
        # Short static text is a label (slot after it) or a tail (slot before it):
        # it must sit where the blueprint put it, not merely occur in the block.
        starts_with_slot = bool(b["slots"]) and b["slots"][0]["start"] <= b["body_start"] + 2
        if ends_in_slot and not starts_with_slot:
            ok = ct[: len(bt)] == bt
        elif starts_with_slot and not ends_in_slot:
            ok = ct[-len(bt) :] == bt
        else:
            ok = ct[: len(bt)] == bt and len(ct) <= len(bt) + 3
        return 1.0 if ok else 0.0
    precision = common / len(ct) if ct else 0.0
    if strict:
        return 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return recall * (0.85 + 0.15 * min(1.0, precision / 0.5))


def _template_regex(b: dict[str, Any]) -> tuple[str, dict[str, dict]]:
    """Regex that a filled copy of blueprint block b must satisfy: its static words
    in order, slots as wildcards, inline hints optional."""
    text = b["text"]
    marks = sorted(
        [(s["start"], s["end"], s) for s in b["slots"]]
        + [(h["start"], h["end"], None) for h in b["inline_hints"]]
    )
    pattern, last, names = "", b["body_start"], {}
    for k, (a, z, slot) in enumerate(marks):
        if a < last:  # lies inside the list marker
            continue
        pattern += _static_words(text[last:a])
        pattern += r"\s*"
        if slot is None:
            pattern += r"(?:\([^()]*\)|\[[^\[\]]*\])?\s*"
        else:
            pattern += f"(?P<g{k}>.*?)"
            names[f"g{k}"] = slot
        last = z
    if text[last:].strip():
        pattern += _static_words(text[last:])
    else:
        # nothing static after the last slot (inline hints aside): it owns the rest
        pattern += r"\s*$"
    return pattern, names


def _static_words(chunk: str) -> str:
    """Fixed words between blanks, matched whole on a side the template spaces
    off: a value starting "w…" must not lend its first letter to the fixed "w"
    after it and leave the blank looking empty."""
    parts = [re.escape(w) for w in chunk.split()]
    if not parts:
        return ""
    if chunk[:1].isspace():
        parts[0] = r"(?<!\w)" + parts[0]
    if chunk[-1:].isspace():
        parts[-1] += r"(?!\w)"
    return r"\s*".join(parts)


def _slot_values(b: dict[str, Any], c_text: str) -> list[tuple[dict, str, str | None]]:
    """For each slot of blueprint block b, what the candidate text holds there."""
    pattern, names = _template_regex(b)
    try:
        m = re.search(pattern, c_text, re.DOTALL)
    except re.error:
        m = None
    out = []
    for name, slot in names.items():
        if not m:
            out.append((slot, "unresolved", None))
            continue
        value = m.group(name) or ""
        if not value.strip():
            # Two blanks with nothing fixed between them share their text out
            # arbitrarily; an empty share says nothing about the blank. Only
            # dots left behind show one was never filled.
            out.append((slot, "unresolved", None))
        elif _FILL_ONLY.match(value):
            out.append((slot, "to_sign" if slot["type"] == "signature" else "unfilled", None))
        else:
            out.append((slot, "filled", value.strip()))
    return out


def _lis_keep(positions: list[int]) -> set[int]:
    tails: list[int] = []
    prev: list[int | None] = []
    for k, pos in enumerate(positions):
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if positions[tails[mid]] < pos:
                lo = mid + 1
            else:
                hi = mid
        prev.append(tails[lo - 1] if lo else None)
        if lo == len(tails):
            tails.append(k)
        else:
            tails[lo] = k
    keep: set[int] = set()
    k = tails[-1] if tails else None
    while k is not None:
        keep.add(k)
        k = prev[k]
    return keep


def evaluate(bp: dict[str, Any], cand: dict[str, Any], *, layout: bool = True) -> dict[str, Any]:
    """Compare any document with a blueprint: what is missing, reworded, extra,
    still blank, laid out differently, or out of order.

    `layout=False` for a candidate read from plain text: it has no alignment,
    indents or automatic numbering, so every such difference would be invented."""
    skip = {"spacer", "pagebreak"}
    B = [b for b in bp["blocks"] if b["role"] not in skip]
    C = [c for c in cand["blocks"] if c["role"] not in skip]
    n, m = len(B), len(C)
    anchors = [i for i, b in enumerate(B) if b["role"] in _ANCHOR_ROLES and tokens(static_text(b))]
    sims = {}
    for i in anchors:
        for j in range(m):
            if C[j]["role"] in _DEPENDENT:
                continue
            s = _sim(B[i], C[j])
            if s >= CHANGED:
                sims[i, j] = s

    # max-weight monotone matching on MATCH-level similarities
    na = len(anchors)
    best = [[0.0] * (m + 1) for _ in range(na + 1)]
    for ai in range(1, na + 1):
        i = anchors[ai - 1]
        w = len(_expected(B[i])[0])
        for j in range(1, m + 1):
            best[ai][j] = max(best[ai - 1][j], best[ai][j - 1])
            s = sims.get((i, j - 1), 0.0)
            if s >= MATCH and best[ai - 1][j - 1] + s * w > best[ai][j]:
                best[ai][j] = best[ai - 1][j - 1] + s * w
    matched: dict[int, int] = {}
    ai, j = na, m
    while ai > 0 and j > 0:
        i = anchors[ai - 1]
        w = len(_expected(B[i])[0])
        s = sims.get((i, j - 1), 0.0)
        if s >= MATCH and abs(best[ai][j] - (best[ai - 1][j - 1] + s * w)) < 1e-9:
            matched[i] = j - 1
            ai, j = ai - 1, j - 1
        elif abs(best[ai][j] - best[ai - 1][j]) < 1e-9:
            ai -= 1
        else:
            j -= 1

    # optional-ness flows down from a conditional block to the blanks and captions under it
    optional = [bool(b["optional"]) for b in B]
    for i in range(1, n):
        if B[i]["role"] in _DEPENDENT and optional[i - 1]:
            optional[i] = True

    rep: dict[str, Any] = {
        "anchors": len(anchors), "matched": 0, "missing": [], "optional_missing": [],
        "changed": [], "extra": [], "slots": [], "layout": [], "residue": [], "order": [],
        "captions_missing": [],
    }
    used_c = set(matched.values())
    total_words = sum(len(_expected(B[i])[0]) for i in anchors) or 1
    got_words = 0.0
    changed_pairs: dict[int, int] = {}
    for i in anchors:
        b = B[i]
        label = ((b["label"] or "") + " " + static_text(b).strip()).strip()[:70]
        if i in matched:
            rep["matched"] += 1
            c = C[matched[i]]
            got_words += sims[i, matched[i]] * len(_expected(b)[0])
            if b["slots"] and b["alternatives"]:
                for slot in b["slots"]:
                    rep["slots"].append(
                        {"id": slot["id"], "type": slot["type"], "status": "choice", "value": None, "p": b["p"]}
                    )
            elif b["slots"]:
                for slot, status, value in _slot_values(b, c["text"][c["body_start"] :]):
                    rep["slots"].append(
                        {"id": slot["id"], "type": slot["type"], "status": status, "value": value, "p": b["p"]}
                    )
            diffs = []
            if (b["align"] in ("center", "right") or c["align"] in ("center", "right")) and b["align"] != c["align"]:
                diffs.append(f"wyrównanie {b['align']} → {c['align']}")
            if bool(b["label"]) != bool(c["label"]):
                diffs.append(f"numeracja {b['label'] or '—'} → {c['label'] or '—'}")
            elif b["label"] and b["label_fmt"] != c["label_fmt"]:
                diffs.append(f"numeracja {b['label']} → {c['label']}")
            if b["bold"] != c["bold"] and b["role"] in ("title", "clause_heading"):
                diffs.append(f"pogrubienie {b['bold']} → {c['bold']}")
            bl, cl = b["indent"].get("left", 0), c["indent"].get("left", 0)
            same_marker = b["label"] and c["label"] and b["label_src"] != c["label_src"]
            if abs(bl - cl) > 360 and not same_marker:
                diffs.append(f"wcięcie {bl} → {cl} twips")
            if layout and diffs:
                rep["layout"].append({"p": b["p"], "role": b["role"], "diffs": diffs})
        else:
            near = max(
                ((sims[i, j], j) for j in range(m) if (i, j) in sims and j not in used_c),
                default=None,
            )
            if near and near[0] >= CHANGED and near[1] not in changed_pairs.values():
                changed_pairs[i] = near[1]
                rep["changed"].append(
                    {"p": b["p"], "was": label, "now": C[near[1]]["text"].strip()[:70], "sim": round(near[0], 2)}
                )
                got_words += near[0] * len(_expected(b)[0])
            else:
                row = {"p": b["p"], "text": label}
                (rep["optional_missing"] if optional[i] else rep["missing"]).append(row)
    rep["words_recalled"] = round(got_words / total_words, 3)
    used_c |= set(changed_pairs.values())

    # captions + the values that sit above them
    c_hint = {tuple(tokens(c["text"])): j for j, c in enumerate(C) if c["role"] == "hint" and tokens(c["text"])}
    consumed: set[int] = set()
    by_p = {b["p"]: (i, b) for i, b in enumerate(B)}
    for i, b in enumerate(B):
        # the rest of a caption broken over paragraphs goes with its first line
        if b["role"] != "hint" or b["hint_for"] not in by_p:
            continue
        if not b["text"].lstrip().startswith(("(", "[")):
            continue
        ti, target = by_p[b["hint_for"]]
        if target["role"] not in ("blank_line", "signature_line"):
            continue
        j = c_hint.get(tuple(tokens(b["text"])))
        if j is None:
            if not optional[ti]:
                rep["captions_missing"].append(
                    {"p": b["p"], "text": b["text"].strip()[:60], "slots": [s["id"] for s in target["slots"]]}
                )
            continue
        prev = C[j - 1] if j > 0 else None
        for slot in target["slots"]:
            if prev is None or prev["role"] == "hint":
                status, value = "unresolved", None
            elif slot["type"] == "signature":
                status, value = "to_sign", None
            else:
                txt = prev["text"].strip()
                status, value = ("unfilled", None) if _FILL_ONLY.match(txt) else ("filled", re.sub(r"[,;]\s*$", "", txt))
            rep["slots"].append({"id": slot["id"], "type": slot["type"], "status": status, "value": value, "p": target["p"]})
        if prev is not None and prev["role"] != "hint":
            consumed.add(j - 1)

    # blank lines without a caption (multi-line fields): values live between matched anchors
    seq = sorted(matched.items())
    bounds = [(-1, -1)] + seq + [(n, m)]
    for (bi0, cj0), (bi1, cj1) in pairwise(bounds):
        free_b = [
            B[i] for i in range(bi0 + 1, bi1)
            if B[i]["role"] not in ("hint", "signature_line")
            and B[i]["slots"]
            and not tokens(static_text(B[i]))
            and not any(x["role"] == "hint" and x["hint_for"] == B[i]["p"] for x in B)
        ]
        free_c = [
            j for j in range(cj0 + 1, cj1)
            if j not in used_c and j not in consumed and C[j]["role"] not in ("hint",)
        ]
        for k, blank in enumerate(free_b):
            if k < len(free_c):
                consumed.add(free_c[k])
                txt = C[free_c[k]]["text"].strip()
                for slot in blank["slots"]:
                    if not _FILL_ONLY.match(txt):
                        status, value = "filled", txt
                    else:
                        signed = slot["type"] == "signature"
                        status, value = ("to_sign" if signed else "unfilled"), None
                    rep["slots"].append({"id": slot["id"], "type": slot["type"], "status": status, "value": value, "p": blank["p"]})
            elif not optional[B.index(blank)]:
                for slot in blank["slots"]:
                    rep["slots"].append({"id": slot["id"], "type": slot["type"], "status": "unresolved", "value": None, "p": blank["p"]})

    for j, c in enumerate(C):
        if j in used_c or j in consumed or c["role"] in _DEPENDENT:
            continue
        if len(tokens(static_text(c))) >= 4:
            rep["extra"].append({"p": c["p"], "text": c["text"].strip()[:80]})
    for c in C:
        if c["role"] == "hint" and c["hint_kind"] == "instruction":
            rep["residue"].append({"p": c["p"], "text": c["text"].strip()[:80], "why": "uwaga redaktora została w dokumencie"})
        if c["role"] in ("body", "item", "field", "title", "party_alias", "field_header") and any(s["kind"] == "blank" for s in c["slots"]):
            rep["residue"].append({"p": c["p"], "text": c["text"].strip()[:80], "why": "pozostały kropki do wypełnienia"})

    # order: matched blocks that fall outside the longest in-sequence run
    free = []
    for i in anchors:
        scored = [(_sim(B[i], C[j], strict=True), -j, j) for j in range(m) if C[j]["role"] not in _DEPENDENT]
        top = max(scored, default=(0, 0, -1))
        if top[0] >= 0.75:
            free.append((i, top[2]))
    keep = _lis_keep([j for _i, j in free])
    for k, (i, _j) in enumerate(free):
        if k not in keep:
            rep["order"].append({"p": B[i]["p"], "text": static_text(B[i]).strip()[:60]})
    return rep


# -------------------------------------------------------------------- CLI ---

_ROUND_TRIP_KEYS = (
    "align", "indent", "label", "label_fmt", "level", "bold", "italic", "tabs", "page_break_before",
)


def check(source: bytes | str, ir: dict[str, Any]) -> list[str]:
    """What would make `fill` or `evaluate` misbehave on this blueprint (empty = fine)."""
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    fresh = extract(data)["blocks"]
    if len(fresh) != len(ir["blocks"]):
        return [f"DOCX ma {len(fresh)} akapitów, blueprint {len(ir['blocks'])}"]
    problems: list[str] = []
    seen: set[str] = set()
    for b in ir["blocks"]:
        if b["role"] != "spacer" and b["text"] != fresh[b["p"]]["text"]:
            problems.append(f"akapit {b['p']}: tekst różni się od DOCX")
            continue
        if b["hint_for"] is not None and not 0 <= b["hint_for"] < len(fresh):
            problems.append(f"akapit {b['p']}: hint_for wskazuje nieistniejący akapit")
        for s in b["slots"]:
            where = f"pole „{s['id']}” (akapit {b['p']})"
            if not re.fullmatch(r"[a-z][a-z0-9_]*", s["id"]):
                problems.append(f"{where}: id tylko z małych liter, cyfr i _")
            if s["id"] in seen:
                problems.append(f"{where}: id użyte drugi raz")
            seen.add(s["id"])
            spans = _find_blanks(b["text"][s["start"] : s["end"]])
            if len(spans) != 1 or spans[0][:2] != (0, s["end"] - s["start"]):
                problems.append(f"{where}: zakres nie obejmuje całych kropek ani [placeholdera]")
    if problems:
        return problems

    # fill every field with a marker and read the file back: only the fields may change
    values = {s["id"]: f"«{s['id']}»" for b in ir["blocks"] for s in b["slots"]}
    filled = extract(fill(data, ir, values))["blocks"]
    for b, before, after in zip(ir["blocks"], fresh, filled, strict=True):
        expected = before["text"]
        for s in sorted(b["slots"], key=lambda s: -s["start"]):
            expected = expected[: s["start"]] + values[s["id"]] + expected[s["end"] :]
        if after["text"] != expected:
            problems.append(f"akapit {b['p']}: po wypełnieniu zmienił się tekst poza polami")
        changed = [k for k in _ROUND_TRIP_KEYS if before[k] != after[k]]
        if changed:
            problems.append(f"akapit {b['p']}: po wypełnieniu zmienił się układ ({', '.join(changed)})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m humanize_pl.layout_ir",
        description="Wzór dokumentu: DOCX z układem i polami do wypełnienia.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    outline = sub.add_parser("outline", help="pokaż, jak narzędzie czyta wzór DOCX")
    outline.add_argument("docx")
    extract_cmd = sub.add_parser("extract", help="wygeneruj szkic blueprintu (YAML) z wzoru DOCX")
    extract_cmd.add_argument("docx")
    extract_cmd.add_argument("--category", required=True, help="np. umowa_o_prace")
    extract_cmd.add_argument("-o", "--output", help="plik YAML (domyślnie stdout)")
    check_cmd = sub.add_parser("check", help="sprawdź blueprint względem jego DOCX")
    check_cmd.add_argument("blueprint", help="plik wzoru .yaml; DOCX z klucza `template` leży obok")
    args = parser.parse_args(argv)

    if args.command == "outline":
        sys.stdout.write(marked(extract(args.docx)) + "\n")
        return 0

    if args.command == "extract":
        ir = extract(args.docx)
        text = to_yaml(ir, template=Path(args.docx).name, category=args.category)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        slots = [s for b in ir["blocks"] for s in b["slots"]]
        unnamed = sum(1 for s in slots if _POSITIONAL_ID.fullmatch(s["id"]))
        sys.stderr.write(
            f"akapitów: {len(ir['blocks'])}, pól: {len(slots)}, "
            f"do nazwania (id pNN_k): {unnamed}. Przejrzyj plik wg README.md.\n"
        )
        return 0

    path = Path(args.blueprint)
    ir = from_yaml(path.read_text(encoding="utf-8"))
    if not ir["template"]:
        sys.stderr.write("BŁĄD: wzór nie wskazuje pliku DOCX (klucz `template`).\n")
        return 1
    problems = check(path.parent / ir["template"], ir)
    for problem in problems:
        sys.stderr.write(f"BŁĄD: {problem}\n")
    if not problems:
        unnamed = [
            s["id"] for b in ir["blocks"] for s in b["slots"] if _POSITIONAL_ID.fullmatch(s["id"])
        ]
        if unnamed:
            sys.stderr.write(f"uwaga: pola z id do zmiany nazwy: {', '.join(unnamed)}\n")
        sys.stderr.write("OK\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
