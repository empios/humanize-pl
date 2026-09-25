from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import regex as re
import typer
from rich import print

from humanize_pl.config import Engine, Mode
from humanize_pl.detect import detect_document
from humanize_pl.document import HumanizeTrack, RewriteBackend
from humanize_pl.flow import FlowResult, humanize
from humanize_pl.io.atomic import write_text_atomic
from humanize_pl.io.docx_structure import inventory_docx
from humanize_pl.safety.protectors import protect_text
from humanize_pl.safety.validators import has_stranded_relative_clause, legal_sensitive_inventory

DEFAULT_MANIFEST = Path("docs_tests/ai_generated/manifest.json")
DEFAULT_OUTPUT = Path("docs_tests/results/latest")
DEFAULT_ENGINES = "basic,hybrid"

app = typer.Typer(add_completion=False, help="Benchmark humanize-pl on legal AI fixtures.")


@dataclass(frozen=True)
class BenchmarkDocument:
    id: str
    path: Path
    type: str = "unknown"
    focus: list[str] = field(default_factory=list)
    source_kind: str = "txt"
    lawyer_path: Path | None = None
    track: HumanizeTrack = HumanizeTrack.legal


@dataclass
class BenchmarkRow:
    document_id: str
    document_type: str
    engine: str
    mode: str
    status: str
    source_path: str
    output_path: str | None = None
    report_path: str | None = None
    accepted_changes: int = 0
    rejected_candidates: int | None = None
    skipped_sentences: int | None = None
    all_candidates: int | None = None
    processing_seconds: float = 0.0
    changes_per_1000_words: float = 0.0
    average_accepted_risk: float = 0.0
    model_status: dict[str, str] = field(default_factory=dict)
    semantic_model: str | None = None
    fluency_model: str | None = None
    operation_types: dict[str, int] = field(default_factory=dict)
    gate_rejections: dict[str, int] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    evaluation: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    track: str = "legal"
    readiness_status: str | None = None


@app.command()
def main(
    manifest: Path = typer.Option(DEFAULT_MANIFEST, "--manifest", help="Benchmark manifest JSON"),
    output: Path = typer.Option(DEFAULT_OUTPUT, "--output", "-o", help="Output directory"),
    engines: str = typer.Option(DEFAULT_ENGINES, "--engines", help="Comma-separated engines"),
    mode: Mode = typer.Option(Mode.standard, "--mode", help="conservative, standard, strong"),
    offline_models: bool = typer.Option(True, "--offline-models/--online-models", help="Use local model cache only"),
    require_models: bool = typer.Option(False, "--require-models", help="Fail if requested models are unavailable"),
    allow_fallback: bool = typer.Option(False, "--allow-fallback", help="Allow nlp/hybrid to fall back to available layers"),
    fail_on_status: bool = typer.Option(
        False,
        "--fail-on-status",
        help="Exit with code 1 when any run is not ok",
    ),
    include_docx: list[Path] = typer.Option(
        [],
        "--include-docx",
        help="Additional DOCX file to benchmark",
    ),
    rewrite_backend: RewriteBackend = typer.Option(RewriteBackend.rules, "--rewrite-backend"),
    require_llm: bool = typer.Option(False, "--require-llm"),
) -> None:
    documents = load_manifest(manifest)
    documents.extend(_docx_documents(include_docx))
    selected_engines = parse_engines(engines)
    rows = run_benchmark(
        documents,
        output_dir=output,
        engines=selected_engines,
        mode=mode,
        offline_models=offline_models,
        require_models=require_models,
        allow_fallback=allow_fallback,
        rewrite_backend=rewrite_backend,
        require_llm=require_llm,
    )
    write_summary_artifacts(rows, output)
    print(f"[green]Benchmark zapisany:[/green] {output}")
    print(f"Dokumenty: {len(documents)}")
    print(f"Uruchomienia: {len(rows)}")
    failed = sum(1 for row in rows if row.status != "ok")
    if failed:
        print(f"[yellow]Statusy wymagające uwagi:[/yellow] {failed}")
        if fail_on_status:
            raise typer.Exit(1)


def parse_engines(value: str) -> list[Engine]:
    engines: list[Engine] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        engines.append(Engine(item))
    if not engines:
        raise ValueError("at least one engine is required")
    return engines


def load_manifest(path: str | Path) -> list[BenchmarkDocument]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    documents: list[BenchmarkDocument] = []
    for item in data:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(item.get("id", ""))):
            raise ValueError("Identyfikator benchmarku może zawierać tylko litery ASCII, cyfry, '_' i '-'.")
        file_name = item["file"]
        source_path = path.parent / file_name
        if not source_path.exists():
            raise FileNotFoundError(f"Manifest document not found: {source_path}")
        lawyer_path = path.parent / item["lawyer_file"] if item.get("lawyer_file") else None
        if lawyer_path is not None and not lawyer_path.exists():
            raise FileNotFoundError(f"Lawyer reference not found: {lawyer_path}")
        documents.append(
            BenchmarkDocument(
                id=item["id"],
                path=source_path,
                type=item.get("type", "unknown"),
                focus=list(item.get("focus", [])),
                source_kind="docx" if source_path.suffix.lower() == ".docx" else "txt",
                lawyer_path=lawyer_path,
                track=HumanizeTrack(item.get("track", "legal")),
            )
        )
    if len({document.id for document in documents}) != len(documents):
        raise ValueError("Powtórzone identyfikatory benchmarku.")
    return sorted(documents, key=lambda doc: doc.id)


def run_benchmark(
    documents: list[BenchmarkDocument],
    *,
    output_dir: Path,
    engines: list[Engine],
    mode: Mode,
    offline_models: bool,
    require_models: bool,
    allow_fallback: bool,
    rewrite_backend: RewriteBackend = RewriteBackend.rules,
    require_llm: bool = False,
) -> list[BenchmarkRow]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[BenchmarkRow] = []
    for document in sorted(documents, key=lambda doc: doc.id):
        original_text = _read_document_text(document)
        for engine in engines:
            row = _run_one(
                document,
                original_text=original_text,
                output_dir=output_dir,
                engine=engine,
                mode=mode,
                offline_models=offline_models,
                require_models=require_models or (engine != Engine.basic and not allow_fallback),
                rewrite_backend=rewrite_backend,
                require_llm=require_llm,
            )
            rows.append(row)
    return rows


def write_summary_artifacts(rows: list[BenchmarkRow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "summary": _aggregate(rows),
        "rows": [asdict(row) for row in rows],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary_csv(rows, output_dir / "summary.csv")
    (output_dir / "review.md").write_text(render_review_markdown(rows), encoding="utf-8")


def safety_checks(original: str, rewritten: str) -> dict[str, Any]:
    original_protected = protect_text(original)
    rewritten_protected = protect_text(rewritten)
    checks = {
        "no_english_markers": not re.search(
            r"\b(?:surprisingly|actually|basically|honestly|literally|overall|moreover)\b",
            rewritten,
            re.IGNORECASE,
        ),
        "no_bad_split_phrase": "Ponadto za wynagrodzeniem" not in rewritten,
        "no_stranded_relative_clause": not has_stranded_relative_clause(rewritten),
        "no_placeholder_leak": "__PROTECTED_" not in rewritten,
        "numbers_preserved": _numbers(original) == _numbers(rewritten),
        "protected_fragments_preserved": _protected_values(original_protected)
        <= _protected_values(rewritten_protected),
        "legal_sensitive_content_preserved": (
            legal_sensitive_inventory(original) == legal_sensitive_inventory(rewritten)
        ),
    }
    checks["passed"] = all(checks.values())
    return checks


def render_review_markdown(rows: list[BenchmarkRow]) -> str:
    lines: list[str] = []
    lines.append("# humanize-pl Benchmark Review")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    aggregate = _aggregate(rows)
    lines.append(f"- Runs: {aggregate['runs']}")
    lines.append(f"- OK: {aggregate['ok']}")
    lines.append(f"- Failed safety: {aggregate['failed_safety']}")
    lines.append(f"- Failed quality: {aggregate['failed_quality']}")
    lines.append(f"- Model unavailable: {aggregate['model_unavailable']}")
    lines.append(f"- Accepted changes: {aggregate['accepted_changes']}")
    lines.append(f"- Processing seconds: {aggregate['processing_seconds']:.4f}")
    lines.append("")
    lines.append("## Per Document")
    lines.append("")
    lines.append(
        "| Document | Engine | Status | Accepted | Rejected | Risk | Changes/1000 | Time(s) | Safety |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda item: (item.document_id, item.engine)):
        lines.append(
            "| "
            f"{row.document_id} | {row.engine} | {row.status} | {row.accepted_changes} | "
            f"{row.rejected_candidates} | {row.average_accepted_risk:.4f} | "
            f"{row.changes_per_1000_words:.4f} | {row.processing_seconds:.4f} | "
            f"{row.safety.get('passed', False)} |"
        )
    lines.append("")
    lines.append("## Safety Signals")
    lines.append("")
    safety_sections = _safety_signal_sections(rows)
    lines.extend(safety_sections or ["Brak sygnałów bezpieczeństwa."])
    lines.append("")
    lines.append("## Four-layer release evaluation")
    lines.append("")
    lines.append(
        "Status OK wymaga łącznie: bezpieczeństwa treści, jakości języka, "
        "zgodności stylu i zachowania formatowania."
    )
    for row in sorted(rows, key=lambda item: (item.document_id, item.engine)):
        layers = row.evaluation or {}
        labels = [
            f"{name}={'OK' if value.get('passed') else 'FAIL'}"
            for name, value in layers.items()
            if isinstance(value, dict)
        ]
        lines.append(f"- {row.document_id}/{row.engine}: " + ", ".join(labels))
    lines.append("")
    lines.append("## Rejected Candidates")
    lines.append("")
    rejected_sections = _rejected_sections(rows)
    lines.extend(rejected_sections or ["Kanoniczny przepływ nie udostępnia pełnego rejestru odrzuconych kandydatów regułowych; brak wpisów nie oznacza zera odrzuceń."])
    lines.append("")
    lines.append("## Needs Review")
    lines.append("")
    needs_review = _needs_review_sections(rows)
    lines.extend(needs_review or ["Brak zmian oznaczonych do ręcznego przeglądu."])
    lines.append("")
    lines.append("## Recommended Next Rules")
    lines.append("")
    lines.extend(_recommended_rules(rows))
    lines.append("")
    return "\n".join(lines)


def _run_one(
    document: BenchmarkDocument,
    *,
    original_text: str,
    output_dir: Path,
    engine: Engine,
    mode: Mode,
    offline_models: bool,
    require_models: bool,
    rewrite_backend: RewriteBackend = RewriteBackend.rules,
    require_llm: bool = False,
) -> BenchmarkRow:
    engine_dir = output_dir / engine.value
    engine_dir.mkdir(parents=True, exist_ok=True)
    row = BenchmarkRow(
        document_id=document.id,
        document_type=document.type,
        engine=engine.value,
        mode=mode.value,
        status="ok",
        source_path=str(document.path),
        track=document.track.value,
    )
    try:
        started_at = perf_counter()
        result, output_path = _process_document(
            document,
            engine_dir=engine_dir,
            engine=engine,
            mode=mode,
            offline_models=offline_models,
            require_models=require_models,
            rewrite_backend=rewrite_backend,
            require_llm=require_llm,
        )
        row.processing_seconds = round(perf_counter() - started_at, 4)
    except RuntimeError as exc:
        row.processing_seconds = round(perf_counter() - started_at, 4)
        row.status = "model_unavailable"
        row.error = str(exc)
        return row

    report_path = engine_dir / f"{document.id}.json"
    write_text_atomic(report_path, json.dumps(result.payload, ensure_ascii=False, indent=2) + "\n")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if result.status != "ok" or not output_path.is_file():
        row.status = "failed_flow"
        row.report_path = str(report_path)
        row.error = "Kanoniczny przepływ nie zapisał poprawnego wyniku."
        return row
    rewritten_text = output_path.read_text(encoding="utf-8") if document.source_kind == "txt" else _read_docx_text(output_path)
    safety = safety_checks(original_text, rewritten_text)
    evaluation = _four_layer_evaluation(
        document,
        original_text=original_text,
        rewritten_text=rewritten_text,
        output_path=output_path,
        safety=safety,
    )
    row.evaluation = evaluation
    item = (payload.get("documents") or [{}])[0]
    row.readiness_status = result.readiness_status
    evaluation["canonical_flow"] = {
        "passed": result.status == "ok" and result.text == rewritten_text
        and payload.get("settings", {}).get("track") == document.track.value
        and bool(result.pdf_report and result.pdf_report.is_file())
        and not payload.get("pdf_error"),
        "readiness_status": result.readiness_status,
        "operations": item.get("operations", {}),
        "scope": "technical_regression_not_human_quality_acceptance",
        "pdf_available": bool(result.pdf_report and result.pdf_report.is_file()),
        "pdf_error": payload.get("pdf_error"),
    }
    if all(layer["passed"] for layer in evaluation.values()):
        status = "ok"
    elif not evaluation["safety"]["passed"]:
        status = "failed_safety"
    else:
        status = "failed_quality"
    return _row_from_payload(
        row,
        payload=payload,
        result=result,
        output_path=output_path,
        report_path=report_path,
        safety=safety,
        status=status,
    )


def _process_document(
    document: BenchmarkDocument,
    *,
    engine_dir: Path,
    engine: Engine,
    mode: Mode,
    offline_models: bool,
    require_models: bool,
    rewrite_backend: RewriteBackend = RewriteBackend.rules,
    require_llm: bool = False,
) -> tuple[FlowResult, Path]:
    output_path = engine_dir / f"{document.id}{'.docx' if document.source_kind == 'docx' else '.txt'}"
    result = humanize(document.path, output=output_path, mode=mode, engine=engine,
                      track=document.track, offline_models=offline_models, require_models=require_models,
                      rewrite_backend=rewrite_backend, require_llm=require_llm, pdf=True,
                      report=engine_dir / f"{document.id}.json")
    return result, output_path


def _row_from_payload(
    row: BenchmarkRow,
    *,
    payload: dict[str, Any],
    result: FlowResult,
    output_path: Path,
    report_path: Path,
    safety: dict[str, Any],
    status: str,
) -> BenchmarkRow:
    row.status = status
    row.output_path = str(output_path)
    row.report_path = str(report_path)
    row.accepted_changes = result.changes_applied
    item = (payload.get("documents") or [{}])[0]
    row.changes_per_1000_words = 1000 * row.accepted_changes / max(1, item.get("words", 0))
    risks = [change["risk"] for change in result.applied_changes if isinstance(change.get("risk"), (float, int))]
    row.average_accepted_risk = mean(risks) if risks else 0.0
    row.model_status = payload.get("layers", {}).get("rewrite", {})
    row.semantic_model = row.model_status.get("semantic_model")
    row.fluency_model = row.model_status.get("fluency_model")
    for change in result.applied_changes:
        issue = change.get("issue", "unknown")
        row.operation_types[issue] = row.operation_types.get(issue, 0) + 1
    row.warnings = result.warnings
    row.safety = safety
    return row


def _four_layer_evaluation(
    document: BenchmarkDocument,
    *,
    original_text: str,
    rewritten_text: str,
    output_path: Path,
    safety: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    before = detect_document(original_text, calibrate_against_default=False)
    after = detect_document(rewritten_text, calibrate_against_default=False)
    tolerance_ok = after.ai_signal_score <= before.ai_signal_score + 0.02
    lawyer_reference = None
    if document.lawyer_path and document.lawyer_path.exists():
        lawyer_reference = document.lawyer_path.read_text(encoding="utf-8")
    language_passed = (
        bool(safety.get("no_english_markers"))
        and bool(safety.get("no_bad_split_phrase"))
        and bool(safety.get("no_stranded_relative_clause"))
    )
    reference_distance = None
    if lawyer_reference is not None:
        lawyer_signal = detect_document(
            lawyer_reference, calibrate_against_default=False
        ).ai_signal_score
        before_distance = abs(before.ai_signal_score - lawyer_signal)
        after_distance = abs(after.ai_signal_score - lawyer_signal)
        reference_distance = {
            "before": round(before_distance, 4),
            "after": round(after_distance, 4),
        }
        language_passed = language_passed and after_distance <= before_distance + 0.02
    language = {
        "passed": language_passed,
        "lawyer_reference_available": lawyer_reference is not None,
        "distance_to_lawyer_style": reference_distance,
    }
    style = {
        "passed": tolerance_ok,
        "signal_before": before.ai_signal_score,
        "signal_after": after.ai_signal_score,
        "maximum_allowed_regression": 0.02,
        "calibrated": False,
    }
    formatting: dict[str, Any] = {"passed": True, "not_applicable": True}
    if document.source_kind == "docx":
        differences = inventory_docx(document.path).structural_differences(
            inventory_docx(output_path)
        )
        formatting = {
            "passed": not differences,
            "not_applicable": False,
            "inventory_differences": differences,
        }
    return {
        "safety": {"passed": bool(safety.get("passed"))},
        "language_quality": language,
        "style_compliance": style,
        "formatting": formatting,
    }


def _write_summary_csv(rows: list[BenchmarkRow], path: Path) -> None:
    fieldnames = [
        "document_id",
        "document_type",
        "engine",
        "mode",
        "status",
        "accepted_changes",
        "rejected_candidates",
        "average_accepted_risk",
        "changes_per_1000_words",
        "processing_seconds",
        "safety_passed",
        "output_path",
        "report_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "document_id": row.document_id,
                    "document_type": row.document_type,
                    "engine": row.engine,
                    "mode": row.mode,
                    "status": row.status,
                    "accepted_changes": row.accepted_changes,
                    "rejected_candidates": row.rejected_candidates,
                    "average_accepted_risk": row.average_accepted_risk,
                    "changes_per_1000_words": row.changes_per_1000_words,
                    "processing_seconds": row.processing_seconds,
                    "safety_passed": row.safety.get("passed"),
                    "output_path": row.output_path,
                    "report_path": row.report_path,
                }
            )


def _aggregate(rows: list[BenchmarkRow]) -> dict[str, Any]:
    return {
        "runs": len(rows),
        "ok": sum(1 for row in rows if row.status == "ok"),
        "failed_safety": sum(1 for row in rows if row.status == "failed_safety"),
        "failed_quality": sum(1 for row in rows if row.status == "failed_quality"),
        "model_unavailable": sum(1 for row in rows if row.status == "model_unavailable"),
        "accepted_changes": sum(row.accepted_changes for row in rows),
        "rejected_candidates": None,  # local candidate counts are not exposed by canonical flow
        "tracks": {track: {"runs": sum(row.track == track for row in rows),
                            "ok": sum(row.track == track and row.status == "ok" for row in rows)}
                   for track in sorted({row.track for row in rows})},
        "processing_seconds": round(sum(row.processing_seconds for row in rows), 4),
        "average_risk": round(
            mean([row.average_accepted_risk for row in rows if row.accepted_changes]),
            4,
        )
        if any(row.accepted_changes for row in rows)
        else 0.0,
    }


def _rejected_sections(rows: list[BenchmarkRow]) -> list[str]:
    sections: list[str] = []
    for row in sorted(rows, key=lambda item: (item.document_id, item.engine)):
        if not row.report_path or not Path(row.report_path).exists():
            continue
        payload = json.loads(Path(row.report_path).read_text(encoding="utf-8"))
        rejected = payload.get("rejected", [])[:5]
        if not rejected:
            continue
        sections.append(f"### {row.document_id} / {row.engine}")
        for item in rejected:
            sections.append(
                f"- `{item.get('rule')}`: {item.get('reason')} "
                f"(sentence={item.get('sentence')})"
            )
    return sections


def _needs_review_sections(rows: list[BenchmarkRow]) -> list[str]:
    sections: list[str] = []
    for row in sorted(rows, key=lambda item: (item.document_id, item.engine)):
        if not row.report_path or not Path(row.report_path).exists():
            continue
        payload = json.loads(Path(row.report_path).read_text(encoding="utf-8"))
        accepted = payload.get("accepted") or [change for item in payload.get("documents", []) for change in item.get("applied_changes", item.get("examples", []))]
        risky = [
            item
            for item in accepted
            if _accepted_item_needs_review(item)
            or (item.get("semantic_similarity") is not None and item["semantic_similarity"] < 0.92)
            or (item.get("fluency_delta") is not None and item["fluency_delta"] < 0)
        ][:5]
        if not risky and row.safety.get("passed", True):
            continue
        sections.append(f"### {row.document_id} / {row.engine}")
        if not row.safety.get("passed", True):
            failed = [key for key, value in row.safety.items() if key != "passed" and not value]
            sections.append(f"- Safety failed: {', '.join(failed)}")
        for item in risky:
            sections.append(
                f"- `{item.get('rule')}` risk={item.get('risk')} "
                f"similarity={item.get('semantic_similarity')} fluency={item.get('fluency_delta')}"
            )
    return sections


def _accepted_item_needs_review(item: dict[str, Any]) -> bool:
    if (item.get("risk") or 0.0) < 0.15:
        return False
    return not (item.get("operation_type") == "ai_artifact_reduction" and _all_gates_passed(item))


def _all_gates_passed(item: dict[str, Any]) -> bool:
    return all(gate.get("ok", True) for gate in item.get("gate_results") or [])


def _safety_signal_sections(rows: list[BenchmarkRow]) -> list[str]:
    sections: list[str] = []
    for row in sorted(rows, key=lambda item: (item.document_id, item.engine)):
        failed_safety = [
            key for key, value in row.safety.items() if key != "passed" and not value
        ]
        gate_failures = {
            key: count
            for key, count in row.gate_rejections.items()
            if key in {
                "no_stranded_relative_clause",
                "finite_verb_presence",
                "sentence_split_safety",
                "known_bad_patterns",
                "no_dangling_connectors",
            }
        }
        if not failed_safety and not gate_failures:
            continue
        sections.append(f"### {row.document_id} / {row.engine}")
        if failed_safety:
            sections.append(f"- Safety failed: {', '.join(failed_safety)}")
        for name, count in sorted(gate_failures.items()):
            sections.append(f"- Gate `{name}` blocked {count} candidate(s)")
    return sections


def _recommended_rules(rows: list[BenchmarkRow]) -> list[str]:
    gate_counts: dict[str, int] = {}
    operation_counts: dict[str, int] = {}
    for row in rows:
        for name, count in row.gate_rejections.items():
            gate_counts[name] = gate_counts.get(name, 0) + count
        for name, count in row.operation_types.items():
            operation_counts[name] = operation_counts.get(name, 0) + count

    recommendations: list[str] = []
    if gate_counts.get("semantic_similarity", 0):
        recommendations.append("- Przejrzeć reguły odrzucane przez `semantic_similarity` i zawęzić ich kontekst.")
    if gate_counts.get("legal_anchor_retention", 0) or gate_counts.get("content_anchor_retention", 0):
        recommendations.append("- Dodać warianty reguł zachowujące kotwice treściowe i prawne.")
    if gate_counts.get("no_stranded_relative_clause", 0):
        recommendations.append("- Przejrzeć nominalizacje blokowane przez `no_stranded_relative_clause`.")
    if operation_counts.get("legal_ai_style_rewrite", 0) < 5:
        recommendations.append("- Rozbudować `legal_ai_style` o kolejne monotonne ramy AI-prawnicze.")
    if not recommendations:
        recommendations.append("- Brak dominującego wzorca; analizować ręcznie sekcję `Needs Review`.")
    return recommendations


def _docx_documents(paths: list[Path]) -> list[BenchmarkDocument]:
    documents: list[BenchmarkDocument] = []
    for path in paths:
        documents.append(
            BenchmarkDocument(
                id=path.stem,
                path=path,
                type="docx",
                focus=["manual docx"],
                source_kind="docx",
            )
        )
    return documents


def _read_document_text(document: BenchmarkDocument) -> str:
    if document.source_kind == "docx":
        return _read_docx_text(document.path)
    return document.path.read_text(encoding="utf-8")


def _read_docx_text(path: Path) -> str:
    from humanize_pl.io.docx_structure import document_text, load_document

    return document_text(load_document(path))


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:[,.]\d+)?", text)


def _protected_values(protected) -> set[str]:
    return set(protected.mapping.values())


if __name__ == "__main__":
    app()
