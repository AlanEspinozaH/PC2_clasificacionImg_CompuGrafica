"""Validación de la exportación y particiones completas por participante."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .catalog import load_catalog

PREPROCESSING = {
    "version": 1, "image_size": 32, "padding": 4, "background": 0, "ink": 1,
    "dtype": "float32", "pixel_range": [0, 1],
}
DATA_FILES = ["X.npy", "y.npy", "participants.npy", "labels.json", "manifest.csv", "metadata.json"]


def catalog_signature(catalog):
    return [
        (s["class_index"], s["id"], s["latex"], s["unicode"])
        for s in catalog["symbols"]
    ]


@dataclass
class PreparedDataset:
    X: np.ndarray
    y: np.ndarray
    participants: np.ndarray
    records: list
    catalog: dict
    fingerprint: str
    excluded_demo: int
    source_type: str = "app_collection"
    source_description: str = "dibujos recolectados en la aplicacion"


def load_dataset(directory, include_demo=False):
    directory = Path(directory)
    if any(not (directory / name).is_file() for name in DATA_FILES):
        raise ValueError("Falta una exportación completa. Pulsa Preparar datos en la aplicación.")
    catalog = load_catalog()
    exported = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    if catalog_signature(exported) != catalog_signature(catalog):
        raise ValueError("El catálogo exportado no corresponde al catálogo actual.")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("preprocessing") != PREPROCESSING or exported.get("image_size") != 32:
        raise ValueError("La convención de preprocesamiento no es compatible.")
    X = np.load(directory / "X.npy", allow_pickle=False)
    y = np.load(directory / "y.npy", allow_pickle=False)
    participants = np.load(directory / "participants.npy", allow_pickle=False)
    if X.ndim != 4 or X.shape[1:] != (32, 32, 1) or X.dtype != np.float32:
        raise ValueError("X debe ser float32 con forma (N, 32, 32, 1).")
    n = len(X)
    if y.shape != (n,) or y.dtype.kind not in "iu":
        raise ValueError("y debe tener una etiqueta entera por imagen.")
    if participants.shape != (n,) or participants.dtype.kind not in "US":
        raise ValueError("Debe existir un código de participante por imagen.")
    if not np.isfinite(X).all() or (n and (X.min() < 0 or X.max() > 1)):
        raise ValueError("X contiene píxeles inválidos; deben estar entre 0 y 1.")
    if n and (y.min() < 0 or y.max() >= len(catalog["symbols"])):
        raise ValueError("Se encontró una etiqueta fuera del catálogo.")
    with (directory / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != n or metadata.get("n_samples") != n:
        raise ValueError("El manifiesto, los metadatos y los arrays tienen tamaños distintos.")
    sample_ids = set()
    hashes = set()
    for i, record in enumerate(records):
        symbol = catalog["symbols"][int(y[i])]
        if (record["class_id"] != symbol["id"] or int(record["class_index"]) != int(y[i])
                or record["participant_id"] != participants[i] or not participants[i].strip()):
            raise ValueError("Las etiquetas o participantes no están alineados con el manifiesto.")
        if record["sample_id"] in sample_ids or record["image_sha256"] in hashes:
            raise ValueError("La exportación contiene muestras repetidas.")
        sample_ids.add(record["sample_id"])
        hashes.add(record["image_sha256"])
    digest = hashlib.sha256()
    for name in DATA_FILES:
        digest.update(name.encode())
        digest.update((directory / name).read_bytes())
    demo = np.char.startswith(participants.astype(str), "DEMO")
    mask = np.ones(n, dtype=bool) if include_demo else ~demo
    return PreparedDataset(
        X[mask], y[mask].astype(np.int64), participants[mask].astype(str),
        [record for record, keep in zip(records, mask) if keep], catalog,
        digest.hexdigest(), 0 if include_demo else int(demo.sum()),
        metadata.get("source_type", "app_collection"), metadata.get("source", "dibujos recolectados en la aplicacion"),
    )


def split_by_participant(dataset, seed=42, validation_participants=None, test_participants=None):
    """Asigna personas enteras; nunca usa las predicciones para elegir una partición."""
    if dataset.source_type == "hasyv2":
        raise ValueError("Los user_id de HASY no identifican personas fiables; usa split_dataset para separar imágenes.")
    groups = sorted(set(dataset.participants.tolist()))
    if len(groups) < 3:
        raise ValueError(
            "Se necesitan al menos 3 personas reales, cada una con las seis categorías, "
            "para entrenamiento, validación y prueba. No cambies el código de una misma persona."
        )
    if (validation_participants is None) != (test_participants is None):
        raise ValueError("Indica conjuntamente los participantes de validación y prueba.")
    if validation_participants is None:
        shuffled = np.random.default_rng(seed).permutation(groups).tolist()
        held_out = max(1, round(len(groups) * 0.15))
        test = set(shuffled[:held_out])
        validation = set(shuffled[held_out:2 * held_out])
    else:
        validation = set(validation_participants)
        test = set(test_participants)
        if not validation or not test or (validation | test) - set(groups):
            raise ValueError("Validación y prueba deben contener participantes presentes en los datos.")
    if validation & test:
        raise ValueError("Una persona no puede pertenecer a validación y prueba.")
    training = set(groups) - validation - test
    result = {}
    for name, members in [("train", training), ("validation", validation), ("test", test)]:
        indices = np.flatnonzero(np.isin(dataset.participants, sorted(members)))
        missing = set(range(len(dataset.catalog["symbols"]))) - set(dataset.y[indices].tolist())
        if missing:
            names = ", ".join(dataset.catalog["symbols"][i]["id"] for i in sorted(missing))
            raise ValueError(f"La partición {name} no tiene ejemplos de: {names}. Revisa la colección.")
        result[name] = indices
    return result


def split_dataset(dataset, seed=42, validation_participants=None, test_participants=None):
    """Usa autores para capturas propias e imágenes estratificadas para HASY."""
    if dataset.source_type != "hasyv2":
        return split_by_participant(dataset, seed, validation_participants, test_participants)
    if validation_participants is not None or test_participants is not None:
        raise ValueError("HASY no permite reservar personas por user_id; su evaluación se separa por imágenes.")
    rng = np.random.default_rng(seed)
    partitions = {name: [] for name in ("train", "validation", "test")}
    for class_index in range(len(dataset.catalog["symbols"])):
        indices = rng.permutation(np.flatnonzero(dataset.y == class_index))
        if len(indices) < 3:
            raise ValueError("HASY necesita al menos tres imágenes por clase para formar las tres particiones.")
        held_out = max(1, round(len(indices) * 0.15))
        partitions["test"].extend(indices[:held_out])
        partitions["validation"].extend(indices[held_out:2 * held_out])
        partitions["train"].extend(indices[2 * held_out:])
    return {name: np.sort(np.asarray(indices, dtype=np.int64)) for name, indices in partitions.items()}
