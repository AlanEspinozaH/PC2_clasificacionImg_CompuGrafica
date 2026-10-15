"""Importa un subconjunto público sin mezclarlo con la recolección de la aplicación."""

import csv
import hashlib
import io
import json
import re
import tarfile
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .catalog import load_catalog
from .dataset import EXPORT_FILES, FIELDS
from .preprocessing import EmptyDrawingError, flatten_on_white, preprocess_image
from .training_data import PREPROCESSING

HASY_URL = "https://zenodo.org/records/259444/files/HASYv2.tar.bz2?download=1"
HASY_DOI = "https://doi.org/10.5281/zenodo.259444"
HASY_MD5 = "fddf23f36e24b5236f6b3a0880c778e3"


def file_digest(path, algorithm="md5"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path, expected_md5=HASY_MD5):
    if file_digest(path) != expected_md5:
        raise ValueError("El archivo HASYv2 no coincide con el checksum publicado. No se importará.")


def download_hasy(path):
    path = Path(path)
    if path.exists():
        verify_archive(path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(HASY_URL, headers={"User-Agent": "PC2-LaTeX-educational-project/1.0"})
    with tempfile.TemporaryDirectory(prefix="download-", dir=path.parent) as temporary:
        staging = Path(temporary) / "HASYv2.tar.bz2"
        with urllib.request.urlopen(request, timeout=30) as response, staging.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 64 * 1024 * 1024:
                    raise ValueError("La descarga supera el tamaño esperado de HASYv2.")
                output.write(chunk)
        verify_archive(staging)
        staging.rename(path)
    return path


def image_key(path):
    """Solo admite nombres de PNG del dataset; nunca extrae rutas arbitrarias."""
    path = PurePosixPath(path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("El archivo contiene una ruta de imagen inválida.")
    if len(path.parts) < 2 or path.parts[-2] != "hasy-data" or not re.fullmatch(r"v2-\d+\.png", path.name):
        raise ValueError("La ruta de una imagen no corresponde al formato HASYv2.")
    return "/".join(path.parts[-2:])


def read_index(archive, catalog):
    tables = {}
    with tarfile.open(archive, mode="r|bz2") as tar:
        for entry in tar:
            name = PurePosixPath(entry.name).name
            if entry.isfile() and name in ("symbols.csv", "hasy-data-labels.csv"):
                if name in tables:
                    raise ValueError("El archivo contiene índices CSV repetidos.")
                content = tar.extractfile(entry).read().decode("utf-8-sig")
                tables[name] = list(csv.DictReader(io.StringIO(content)))
            if len(tables) == 2:
                break
    if len(tables) != 2:
        raise ValueError("Faltan symbols.csv o hasy-data-labels.csv en el archivo.")
    symbols = {row["symbol_id"]: row["latex"] for row in tables["symbols.csv"]}
    selected = {str(s["hasy_symbol_id"]): s for s in catalog["symbols"]}
    for identifier, symbol in selected.items():
        if symbols.get(identifier) != symbol["latex"]:
            raise ValueError(f"El ID HASY {identifier} no corresponde a {symbol['latex']}.")
    index = {}
    for row in tables["hasy-data-labels.csv"]:
        if row["symbol_id"] not in selected:
            continue
        if row["latex"] != selected[row["symbol_id"]]["latex"]:
            raise ValueError("Una etiqueta HASY no corresponde a su ID de símbolo.")
        key = image_key(row["path"])
        if key in index:
            raise ValueError("El índice de HASY contiene una ruta duplicada.")
        index[key] = row
    return index, selected


def prepare_hasy(archive, destination, per_class=800, seed=42, expected_md5=HASY_MD5):
    archive, destination = Path(archive), Path(destination)
    if destination.exists():
        raise ValueError("El destino HASY ya existe. Usa otro --output para conservar el subconjunto.")
    if per_class < 1:
        raise ValueError("El número de muestras por categoría debe ser positivo.")
    verify_archive(archive, expected_md5)
    catalog = load_catalog()
    index, selected = read_index(archive, catalog)
    pools = {symbol["id"]: [] for symbol in catalog["symbols"]}
    seen, conflicts = {}, set()
    skipped = {"blank": 0, "duplicates": 0, "invalid_source_user_id": 0}
    with tarfile.open(archive, mode="r|bz2") as tar:
        for entry in tar:
            if not entry.isfile() or not entry.name.endswith(".png"):
                continue
            key = image_key(entry.name)
            row = index.pop(key, None)
            if row is None:
                continue
            user = row["user_id"]
            if not re.fullmatch(r"\d+", user) or int(user) < 1:
                skipped["invalid_source_user_id"] += 1
                continue
            symbol = selected[row["symbol_id"]]
            with Image.open(io.BytesIO(tar.extractfile(entry).read())) as image:
                if image.format != "PNG" or image.size != (32, 32):
                    raise ValueError("Una imagen HASY no es un PNG de 32 × 32.")
                try:
                    processed = preprocess_image(image)
                except EmptyDrawingError:
                    skipped["blank"] += 1
                    continue
                encoded = io.BytesIO()
                flatten_on_white(image).save(encoded, format="PNG")
            png = encoded.getvalue()
            digest = hashlib.sha256(png).hexdigest()
            if digest in seen:
                skipped["duplicates"] += 1
                if seen[digest] != symbol["id"]:
                    conflicts.add(digest)
                continue
            seen[digest] = symbol["id"]
            pools[symbol["id"]].append({"source": row, "key": key, "png": png,
                                         "pixels": processed, "digest": digest})
    if index:
        raise ValueError("El archivo no contiene todas las imágenes indicadas en su CSV.")
    available = {}
    chosen = []
    rng = np.random.default_rng(seed)
    for symbol in catalog["symbols"]:
        samples = sorted((r for r in pools[symbol["id"]] if r["digest"] not in conflicts), key=lambda r: r["key"])
        available[symbol["id"]] = len(samples)
        if len(samples) < per_class:
            raise ValueError(f"{symbol['id']} tiene {len(samples)} muestras válidas y únicas; pide menos con --per-class.")
        chosen.extend((symbol, samples[i]) for i in rng.permutation(len(samples))[:per_class])
    created_at = datetime.now(timezone.utc).isoformat()
    records, provenance = [], []
    for symbol, item in chosen:
        original = PurePosixPath(item["key"]).stem
        sample_id = "hasyv2-" + original
        record = {"sample_id": sample_id, "path": f"{symbol['id']}/{sample_id}.png",
                  "class_id": symbol["id"], "class_index": symbol["class_index"],
                  "participant_id": "HASY_" + item["source"]["user_id"], "input_method": "unknown",
                  "created_at": created_at, "image_sha256": item["digest"]}
        records.append(record)
        provenance.append(dict(record, source_path=item["key"], source_user_id=item["source"]["user_id"],
                               source_symbol_id=item["source"]["symbol_id"]))
    X = np.stack([item["pixels"] for _symbol, item in chosen])
    y = np.asarray([r["class_index"] for r in records], dtype=np.int64)
    participants = np.asarray([r["participant_id"] for r in records], dtype=str)
    metadata = {
        "created_at": created_at, "n_samples": len(records), "n_participants": len(set(participants.tolist())),
        "class_counts": {s["id"]: per_class for s in catalog["symbols"]}, "X_shape": list(X.shape), "y_shape": list(y.shape),
        "preprocessing": PREPROCESSING, "target_per_class": catalog["target_per_class"],
        "complete_collection": False, "source_type": "hasyv2", "source": "subconjunto del dataset publico HASYv2",
        "source_url": HASY_DOI, "archive_md5": file_digest(archive), "archive_sha256": file_digest(archive, "sha256"),
        "selection_seed": seed, "per_class": per_class, "available_after_filtering": available,
        "skipped": dict(skipped, conflicting_drawings=len(conflicts)),
        "note": "Experimento con datos publicos; no acredita recoleccion propia ni evaluacion en el lienzo.",
        "participant_identity_reliable": False,
        "identity_note": "user_id no equivale a persona: 16925 agrupa contribuciones de Detexify; ver sección III del artículo HASYv2.",
        "identity_reference": "https://arxiv.org/abs/1701.08380",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="import-", dir=destination.parent) as temporary:
        root = Path(temporary) / "dataset"
        raw, processed_root = root / "raw", root / "processed"
        processed_root.mkdir(parents=True)
        for record, (_symbol, item) in zip(records, chosen):
            path = raw / record["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item["png"])
        for path, rows, fields in [(raw / "samples.csv", records, FIELDS),
                                    (processed_root / "manifest.csv", records, FIELDS),
                                    (processed_root / "source_manifest.csv", provenance,
                                     FIELDS + ["source_path", "source_user_id", "source_symbol_id"])]:
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        for name, value in [("X.npy", X), ("y.npy", y), ("participants.npy", participants)]:
            np.save(processed_root / name, value, allow_pickle=False)
        for name, value in [("labels.json", catalog), ("metadata.json", metadata)]:
            (processed_root / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        with zipfile.ZipFile(processed_root / "dataset.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for filename in EXPORT_FILES + ["source_manifest.csv"]:
                bundle.write(processed_root / filename, arcname=filename)
        root.rename(destination)
    return metadata
