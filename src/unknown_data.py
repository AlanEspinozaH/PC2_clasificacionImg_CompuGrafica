"""Separate, append-only collection of drawings outside the six-class catalog."""

import csv
import hashlib
import io
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .dataset import normalize_participant_id
from .preprocessing import flatten_on_white

UNKNOWN_KINDS = {
    "integral": "∫ Integral",
    "sqrt": "√ Raíz cuadrada",
    "plus": "+ Signo más",
    "pi": "π Pi",
    "random": "Garabato aleatorio",
    "other": "Otro dibujo desconocido",
}
REQUIRED_EVALUATION_KINDS = ("integral", "sqrt", "plus", "pi", "random")
UNKNOWN_FIELDS = [
    "sample_id", "path", "unknown_kind", "participant_id",
    "input_method", "created_at", "image_sha256",
]


class DuplicateUnknownDrawingError(ValueError):
    """The exact same image must not be counted twice as an unknown example."""


class UnknownDrawingStore:
    def __init__(self, data_root, *, is_store_root=False):
        base = Path(data_root)
        self.root = base if is_store_root else base / "unknown"
        self.raw_root = self.root / "raw"
        self.manifest = self.root / "samples.csv"
        self.lock = threading.RLock()

    def records(self):
        with self.lock:
            if not self.manifest.exists():
                return []
            with self.manifest.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != UNKNOWN_FIELDS:
                    raise ValueError("El manifiesto de dibujos desconocidos no es válido.")
                records = list(reader)
                sample_ids = set()
                hashes = set()
                for record in records:
                    path_value = record.get("path")
                    sample_id = record.get("sample_id")
                    digest = record.get("image_sha256")
                    if (not isinstance(path_value, str) or not isinstance(sample_id, str)
                            or not isinstance(digest, str)):
                        raise ValueError("El manifiesto contiene una muestra desconocida inválida.")
                    relative_path = Path(path_value)
                    if (record["unknown_kind"] not in UNKNOWN_KINDS
                            or normalize_participant_id(record["participant_id"])
                            != record["participant_id"]
                            or not sample_id
                            or len(digest) != 64
                            or any(character not in "0123456789abcdef" for character in digest)
                            or relative_path.is_absolute()
                            or ".." in relative_path.parts
                            or not relative_path.parts
                            or relative_path.parts[0] != "raw"
                            or record["input_method"] not in ("mouse", "touch", "pen", "unknown")):
                        raise ValueError("El manifiesto contiene una muestra desconocida inválida.")
                    if sample_id in sample_ids or digest in hashes:
                        raise ValueError("El manifiesto contiene muestras desconocidas repetidas.")
                    sample_ids.add(sample_id)
                    hashes.add(digest)
            return records

    def counts(self):
        result = {kind: 0 for kind in UNKNOWN_KINDS}
        for record in self.records():
            result[record["unknown_kind"]] += 1
        return result

    def _write_records(self, records):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", newline="", encoding="utf-8", dir=self.root,
                prefix=".samples-", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=UNKNOWN_FIELDS)
                writer.writeheader()
                writer.writerows(records)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.manifest)
        except OSError:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise

    def save(self, image, unknown_kind, participant_id, input_method):
        if unknown_kind not in UNKNOWN_KINDS:
            raise ValueError("Selecciona una categoría de dibujo desconocido válida.")
        participant_id = normalize_participant_id(participant_id)
        if input_method not in ("mouse", "touch", "pen", "unknown"):
            raise ValueError("El método de entrada no es válido.")
        output = io.BytesIO()
        flatten_on_white(image).save(output, format="PNG")
        image_bytes = output.getvalue()
        digest = hashlib.sha256(image_bytes).hexdigest()

        with self.lock:
            records = self.records()
            if any(record["image_sha256"] == digest for record in records):
                raise DuplicateUnknownDrawingError(
                    "Este dibujo desconocido ya fue guardado. Dibuja una muestra nueva."
                )
            self.root.mkdir(parents=True, exist_ok=True)
            category_root = self.raw_root / unknown_kind
            category_root.mkdir(parents=True, exist_ok=True)
            destination = None
            for _attempt in range(100):
                sample_id = uuid.uuid4().hex
                relative_path = Path("raw") / unknown_kind / f"{participant_id}_{sample_id}.png"
                candidate = self.root / relative_path
                try:
                    with candidate.open("xb") as stream:
                        stream.write(image_bytes)
                    destination = candidate
                    break
                except FileExistsError:
                    continue
                except OSError:
                    candidate.unlink(missing_ok=True)
                    raise
            if destination is None:
                raise OSError("No se pudo generar un identificador único para la muestra.")

            record = {
                "sample_id": sample_id,
                "path": relative_path.as_posix(),
                "unknown_kind": unknown_kind,
                "participant_id": participant_id,
                "input_method": input_method,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "image_sha256": digest,
            }
            try:
                self._write_records([*records, record])
            except OSError:
                destination.unlink(missing_ok=True)
                raise
        return record
