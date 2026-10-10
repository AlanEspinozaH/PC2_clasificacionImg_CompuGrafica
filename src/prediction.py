"""Carga diferida: la recolección no necesita tener instalado PyTorch."""

import json
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
                if info.get("purpose") not in ("pilot", "collected_dataset"):
                    raise ValueError("Falta identificar el tipo de experimento del modelo.")
                model = SymbolCNN(len(self.catalog["symbols"]))
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
                    raise ValueError("El modelo contiene parámetros no finitos.")
                model.eval()
                self.model, self.info = model, info
            except (OSError, ValueError, KeyError, RuntimeError, TypeError, AttributeError,
                    IndexError, EOFError, pickle.UnpicklingError) as error:
                raise ModelUnavailableError(f"No se pudo cargar el modelo: {error}") from error

    def status(self):
        try:
            self.load()
            return {"ready": True, "pilot": self.info["purpose"] == "pilot",
                    "message": "Modelo piloto: comprueba las sugerencias." if self.info["purpose"] == "pilot"
                    else "Dibuja uno de los seis símbolos del catálogo para consultar su comando."}
        except ModelUnavailableError as error:
            return {"ready": False, "pilot": False, "message": str(error)}

    def predict(self, processed):
        self.load()
        import torch
        tensor = torch.from_numpy(np.ascontiguousarray(processed.transpose(2, 0, 1)[None]))
        with self.lock, torch.inference_mode():
            probabilities = self.model(tensor).softmax(1)[0].numpy()
        if not np.isfinite(probabilities).all():
            raise ModelUnavailableError("El modelo produjo valores inválidos.")
        ranking = np.argsort(-probabilities, kind="stable")[:3]
        candidates = []
        for index in ranking:
            symbol = self.catalog["symbols"][index]
            candidates.append({"class_id": symbol["id"], "class_index": int(index),
                               "name": symbol["name"], "unicode": symbol["unicode"],
                               "latex": symbol["latex"], "example_latex": symbol["example_latex"],
                               "score": float(probabilities[index])})
        return {"candidates": candidates, "pilot": self.info["purpose"] == "pilot",
                "message": "Sugerencias entre las seis categorías. Verifica el símbolo antes de copiar el comando."}
