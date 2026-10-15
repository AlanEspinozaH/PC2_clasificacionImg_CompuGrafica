"""Métricas con un orden fijo de clases y sin alterar las imágenes de prueba."""

import numpy as np


def classification_metrics(y, probabilities, catalog):
    y = np.asarray(y)
    probabilities = np.asarray(probabilities)
    n_classes = len(catalog["symbols"])
    if not len(y) or probabilities.shape != (len(y), n_classes):
        raise ValueError("Las probabilidades deben tener una fila por muestra y una columna por clase.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Se encontraron probabilidades no finitas.")
    ranking = np.argsort(-probabilities, axis=1, kind="stable")
    predicted = ranking[:, 0]
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y, predicted), 1)
    support = confusion.sum(axis=1)
    selected = confusion.sum(axis=0)
    correct = np.diag(confusion)
    precision = np.divide(correct, selected, out=np.zeros(n_classes), where=selected != 0)
    recall = np.divide(correct, support, out=np.zeros(n_classes), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(n_classes), where=precision + recall != 0)
    return {
        "n_samples": len(y), "top_1_accuracy": float(np.mean(predicted == y)),
        "top_3_accuracy": float(np.mean(np.any(ranking[:, :3] == y[:, None], axis=1))),
        "macro_f1": float(f1.mean()), "confusion_matrix": confusion.tolist(),
        "confusion_axes": {"rows": "real", "columns": "predicted"},
        "class_order": [s["id"] for s in catalog["symbols"]],
        "per_class": [
            {"id": s["id"], "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i]), "support": int(support[i])}
            for i, s in enumerate(catalog["symbols"])
        ],
        "uniform_random_baseline": {"top_1_accuracy": 1 / n_classes,
                                    "top_3_accuracy": min(3, n_classes) / n_classes},
    }
