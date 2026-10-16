"""Carga diferida: la recolección no necesita tener instalado PyTorch."""

import json
import hashlib
import pickle
import threading
from pathlib import Path

import numpy as np

from .training_data import PREPROCESSING, catalog_signature


class ModelUnavailableError(ValueError):
    pass


class PredictionService:
    def __init__(self, model_root, catalog):
        self.root = Path(model_root)
        self.catalog = catalog
        self.model = None
        self.info = None
        self.energy_threshold = None
        self.lock = threading.RLock()

    def load(self):
        with self.lock:
            if self.model is not None:
                return
            if not (self.root / "model.pt").is_file():
                raise ModelUnavailableError("Todavía no hay un modelo entrenado. Puedes continuar recolectando dibujos.")
            try:
                import torch
                from .model import SymbolCNN
            except ImportError as error:
                raise ModelUnavailableError("Falta instalar PyTorch con requirements-cpu.txt.") from error
            try:
                info = json.loads((self.root / "model_info.json").read_text(encoding="utf-8"))
                checkpoint = torch.load(self.root / "model.pt", map_location="cpu", weights_only=True)
                for description in (info, checkpoint):
                    if (description.get("schema_version") != 1
                            or description.get("architecture") != SymbolCNN.architecture
                            or description.get("preprocessing") != PREPROCESSING
                            or catalog_signature(description["catalog"]) != catalog_signature(self.catalog)):
                        raise ValueError("El modelo no coincide con el catálogo o procesamiento actual.")
                if info.get("purpose") not in ("pilot", "collected_dataset", "external_baseline"):
                    raise ValueError("Falta identificar el tipo de experimento del modelo.")
                model = SymbolCNN(len(self.catalog["symbols"]))
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
                    raise ValueError("El modelo contiene parámetros no finitos.")
                model.eval()
                calibration_path = self.root / "rejection_calibration.json"
                if calibration_path.is_file():
                    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                    checkpoint_digest = hashlib.sha256(
                        (self.root / "model.pt").read_bytes()
                    ).hexdigest()
                    threshold = calibration.get("energy_threshold")
                    if (calibration.get("schema_version") != 1
                            or calibration.get("method") != "energy"
                            or calibration.get("model_sha256") != checkpoint_digest
                            or (info.get("dataset_sha256") is not None
                                and calibration.get("dataset_sha256")
                                != info.get("dataset_sha256"))
                            or isinstance(threshold, bool)
                            or not isinstance(threshold, (int, float))
                            or not np.isfinite(threshold)):
                        raise ValueError(
                            "La calibración de rechazo no corresponde a este modelo."
                        )
                    self.energy_threshold = float(threshold)
                self.model, self.info = model, info
            except (OSError, ValueError, KeyError, RuntimeError, TypeError, AttributeError,
                    IndexError, EOFError, pickle.UnpicklingError) as error:
                raise ModelUnavailableError(f"No se pudo cargar el modelo: {error}") from error

    def status(self):
        try:
            self.load()
            preliminary = self.info["purpose"] != "collected_dataset"
            if self.energy_threshold is None:
                message = (
                    "Rechazo preliminar sin calibrar: Softmax no descarta por sí solo "
                    "todos los dibujos desconocidos."
                )
                method = "softmax_confidence_margin"
            else:
                message = "Rechazo por energía calibrado con un conjunto de validación."
                method = "energy"
            if preliminary:
                message = "Modelo de prueba: comprueba las sugerencias. " + message
            return {
                "ready": True,
                "pilot": preliminary,
                "rejection_calibrated": self.energy_threshold is not None,
                "rejection_method": method,
                "message": message,
            }
        except ModelUnavailableError as error:
            return {"ready": False, "pilot": False, "message": str(error)}

    def logits(self, processed):
        self.load()
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(processed.transpose(2, 0, 1)[None]))
        with self.lock, torch.inference_mode():
            logits = self.model(tensor)[0].numpy()
        if not np.isfinite(logits).all():
            raise ModelUnavailableError("El modelo produjo valores inválidos.")
        return logits

    def predict(self, processed):
        self.load()
        from .rejection import energy_scores

        logits = self.logits(processed)
        shifted = logits - logits.max()
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum()
        energy = float(energy_scores(logits[None])[0])
        if not np.isfinite(probabilities).all():
            raise ModelUnavailableError("El modelo produjo valores inválidos.")
        ranking = np.argsort(-probabilities, kind="stable")[:3]
        top1 = float(probabilities[ranking[0]])
        top2 = float(probabilities[ranking[1]])
        margin = top1 - top2
        if self.energy_threshold is None:
            is_confident = top1 >= 0.80 and margin >= 0.20
            rejection_method = "softmax_confidence_margin"
            calibration_status = "uncalibrated"
        else:
            is_confident = energy <= self.energy_threshold
            rejection_method = "energy"
            calibration_status = "calibrated"

        candidates = []
        if is_confident:
            for index in ranking:
                symbol = self.catalog["symbols"][index]
                candidates.append({
                    "class_id": symbol["id"], "class_index": int(index),
                    "name": symbol["name"], "unicode": symbol["unicode"],
                    "latex": symbol["latex"], "example_latex": symbol["example_latex"],
                    "score": float(probabilities[index]),
                })

        return {
            "candidates": candidates,
            "pilot": self.info["purpose"] != "collected_dataset",
            "recognized": bool(is_confident),
            "score": top1,
            "margin": margin,
            "energy": energy,
            "rejection_method": rejection_method,
            "calibration_status": calibration_status,
            "message": (
                f"Símbolo reconocido: {candidates[0]['unicode']} — {candidates[0]['name']}"
                if is_confident else
                "Sin coincidencia confiable. No podemos identificar este dibujo como uno "
                "de los seis símbolos admitidos. Intenta dibujarlo nuevamente o consulta "
                "las categorías disponibles."
            ),
        }
