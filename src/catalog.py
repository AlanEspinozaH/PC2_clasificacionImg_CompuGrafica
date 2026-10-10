"""El orden del catálogo determina las etiquetas de la futura CNN."""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_catalog():
    path = PROJECT_ROOT / "config" / "symbols.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    symbols = catalog["symbols"]
    indices = [symbol["class_index"] for symbol in symbols]
    identifiers = [symbol["id"] for symbol in symbols]
    if not symbols or indices != list(range(len(symbols))):
        raise ValueError("Las clases deben estar ordenadas con índices consecutivos desde 0.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Los identificadores del catálogo deben ser únicos.")
    if not all(re.fullmatch(r"[a-z][a-z0-9_]*", value) for value in identifiers):
        raise ValueError("Los identificadores deben usar letras minúsculas, números o guiones bajos.")
    return catalog
