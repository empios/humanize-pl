"""Local diagnostics without network probes, downloads or secret values."""

from __future__ import annotations

import platform
import shutil
from importlib.metadata import PackageNotFoundError, version

from humanize_pl.llm import LlmConfigurationError, LlmSettings
from humanize_pl.privacy import processing_location


def diagnose() -> dict:
    packages = {}
    for name in ('humanize-pl', 'python-docx', 'openpyxl', 'gradio', 'reportlab', 'morfeusz2',
                 'stanza', 'torch', 'transformers', 'sentence-transformers'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    model = {'configured': False, 'probed': False}
    try:
        settings = LlmSettings.from_environment()
    except LlmConfigurationError as exc:
        model['note'] = str(exc)
    else:
        model.update(configured=True, model=settings.model, location=processing_location(settings.endpoint),
                     timeout_seconds=settings.timeout_seconds)
    return {'python': platform.python_version(), 'platform': platform.system(), 'packages': packages,
            'renderer_on_path': bool(shutil.which('soffice') or shutil.which('libreoffice')),
            'pdf_rasterizer_on_path': bool(shutil.which('pdftoppm')), 'hosted_model': model,
            'local_model_weights': 'not_loaded_not_verified',
            'note': 'Obecność pakietu nie potwierdza dostępności wag. Diagnostyka nie pobiera modeli ani nie wysyła zapytań.'}
