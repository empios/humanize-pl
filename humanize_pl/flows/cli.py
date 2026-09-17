"""Compatibility module for flows CLI.

All functionality is now unified under `humanize_pl.cli.app` (the `humanize-pl` command).
This module re-exports the unified app and its commands for backward compatibility.
"""

from __future__ import annotations

from humanize_pl.cli import (
    app,
    blueprint_command,
    docx_command,
    nli_command,
    profile_command,
    report_command,
    run_command,
    xlsx_command,
    ui_command,
)

__all__ = [
    "app",
    "blueprint_command",
    "docx_command",
    "nli_command",
    "profile_command",
    "report_command",
    "run_command",
    "ui_command",
    "xlsx_command",
]

if __name__ == "__main__":
    app()
