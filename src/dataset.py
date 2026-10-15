"""PNG originales, metadatos CSV y una exportación reproducible para entrenar."""

import csv
import hashlib
import io
import json
import os
import tempfile
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .preprocessing import flatten_on_white, preprocess_image

FIELDS = [
    "sample_id", "path", "class_id", "class_index", "participant_id",
    "input_method", "created_at", "image_sha256",
]
EXPORT_FILES = [
    "X.npy", "y.npy", "participants.npy", "labels.json", "manifest.csv", "metadata.json",
]


class DuplicateDrawingError(ValueError):
    """Guardar otra vez el mismo PNG no aporta un dibujo nuevo."""


class DatasetStore:
    def __init__(self, data_root, catalog):
        self.data_root = Path(data_root)
        self.raw_root = self.data_root / "raw"
        self.processed_root = self.data_root / "processed"
        self.manifest = self.raw_root / "samples.csv"
        self.catalog = catalog
        self.symbols = {symbol["id"]: symbol for symbol in catalog["symbols"]}
        self.lock = threading.RLock()
        self.export_lock = threading.Lock()

    def records(self):
        with self.lock:
            if not self.manifest.exists():
                return []
            with self.manifest.open(newline="", encoding="utf-8") as stream:
                return list(csv.DictReader(stream))

    def counts(self):
        counts = {identifier: 0 for identifier in self.symbols}
        for record in self.records():
            counts[record["class_id"]] += 1
        return counts

    def save(self, image, class_id, participant_id, input_method):
        symbol = self.symbols[class_id]
        png = io.BytesIO()
        flatten_on_white(image).save(png, format="PNG")
        image_bytes = png.getvalue()
        digest = hashlib.sha256(image_bytes).hexdigest()
        sample_id = uuid.uuid4().hex
        relative_path = Path(class_id) / f"{sample_id}.png"
        record = {
            "sample_id": sample_id,
            "path": relative_path.as_posix(),
            "class_id": class_id,
            "class_index": symbol["class_index"],
            "participant_id": participant_id,
            "input_method": input_method,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image_sha256": digest,
        }
        with self.lock:
            if any(row["image_sha256"] == digest for row in self.records()):
                raise DuplicateDrawingError(
                    "Este dibujo ya fue guardado. Borra el lienzo y dibuja una muestra nueva."
                )
            destination = self.raw_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(image_bytes)
            try:
                new_manifest = not self.manifest.exists()
                with self.manifest.open("a", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    if new_manifest:
                        writer.writeheader()
                    writer.writerow(record)
            except OSError:
                destination.unlink(missing_ok=True)
                raise
        return record

    def export(self):
        with self.export_lock:
            records = self.records()
            if not records:
                raise ValueError("Guarda al menos un dibujo antes de preparar los datos.")
            images = []
            for record in records:
                expected_index = self.symbols[record["class_id"]]["class_index"]
                if int(record["class_index"]) != expected_index:
                    raise ValueError("El catálogo cambió: revisa las etiquetas de las muestras.")
                with Image.open(self.raw_root / record["path"]) as image:
                    images.append(preprocess_image(image, size=self.catalog["image_size"]))
            X = np.stack(images)
            y = np.asarray([int(row["class_index"]) for row in records], dtype=np.int64)
            participants = np.asarray([row["participant_id"] for row in records], dtype=str)
            counts = {
                key: sum(row["class_id"] == key for row in records) for key in self.symbols
            }
            metadata = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "n_samples": len(records),
                "n_participants": len(set(participants.tolist())),
                "class_counts": counts,
                "X_shape": list(X.shape),
                "y_shape": list(y.shape),
                "preprocessing": {
                    "version": 1, "image_size": self.catalog["image_size"],
                    "padding": 4, "background": 0, "ink": 1,
                    "dtype": "float32", "pixel_range": [0, 1],
                },
                "target_per_class": self.catalog["target_per_class"],
                "complete_collection": all(
                    count >= self.catalog["target_per_class"] for count in counts.values()
                ),
                "source": "dibujos recolectados en la aplicacion",
            }
            self.data_root.mkdir(parents=True, exist_ok=True)
            self.processed_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="export-", dir=self.data_root) as temporary:
                staging = Path(temporary)
                np.save(staging / "X.npy", X, allow_pickle=False)
                np.save(staging / "y.npy", y, allow_pickle=False)
                np.save(staging / "participants.npy", participants, allow_pickle=False)
                (staging / "labels.json").write_text(
                    json.dumps(self.catalog, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (staging / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                with (staging / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(records)
                with zipfile.ZipFile(
                    staging / "dataset.zip", "w", compression=zipfile.ZIP_DEFLATED
                ) as archive:
                    for filename in EXPORT_FILES:
                        archive.write(staging / filename, arcname=filename)
                for filename in EXPORT_FILES + ["dataset.zip"]:
                    os.replace(staging / filename, self.processed_root / filename)
            return metadata
