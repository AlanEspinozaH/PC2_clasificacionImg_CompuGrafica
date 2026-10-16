"""Validation-only calibration and reporting for closed-set rejection strategies."""

import numpy as np

from .evaluation import classification_metrics


def energy_scores(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Los logits deben ser una matriz finita con al menos una muestra.")
    maximum = values.max(axis=1)
    return -(maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1)))


def calibrate_energy_threshold(known_logits, unknown_logits, known_groups=None,
                               unknown_groups=None):
    known = energy_scores(known_logits)
    unknown = energy_scores(unknown_logits)
    if not len(known) or not len(unknown):
        raise ValueError("La validación necesita muestras conocidas y desconocidas.")
    if ((known_groups is not None and len(known_groups) != len(known))
            or (unknown_groups is not None and len(unknown_groups) != len(unknown))):
        raise ValueError("Los grupos de validación no están alineados con los logits.")

    def group_mean(values, groups):
        if groups is None:
            return float(np.mean(values))
        groups = np.asarray(groups, dtype=str)
        return float(np.mean([
            np.mean(values[groups == group]) for group in sorted(set(groups.tolist()))
        ]))

    joined = np.concatenate((known, unknown))
    ordered = np.sort(joined)
    gaps = ordered[1:] - ordered[:-1]
    midpoints = ordered[:-1] + gaps / 2
    margin = max(1e-6, float(np.max(np.abs(ordered))) * 1e-6)
    thresholds = np.concatenate((
        [ordered[0] - margin], midpoints, [ordered[-1] + margin],
    ))

    best = None
    for threshold in thresholds:
        known_rejected = known > threshold
        unknown_rejected = unknown > threshold
        known_acceptance = group_mean(~known_rejected, known_groups)
        unknown_rejection = group_mean(unknown_rejected, unknown_groups)
        balanced_accuracy = (known_acceptance + unknown_rejection) / 2
        candidate = (balanced_accuracy, known_acceptance, float(threshold),
                     known_acceptance, unknown_rejection)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    return {
        "method": "energy",
        "threshold": best[2],
        "validation_balanced_accuracy": best[0],
        "validation_known_acceptance_rate": best[3],
        "validation_unknown_rejection_rate": best[4],
        "validation_known_count": int(len(known)),
        "validation_unknown_count": int(len(unknown)),
    }


def rejection_metrics(y_known, known_scores, known_probabilities, unknown_scores,
                      unknown_kinds, catalog, method, threshold=None,
                      unknown_probabilities=None):
    y_known = np.asarray(y_known, dtype=np.int64)
    known_scores = (
        None if known_scores is None else np.asarray(known_scores, dtype=np.float64)
    )
    known_probabilities = np.asarray(known_probabilities, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    unknown_kinds = np.asarray(unknown_kinds, dtype=str)
    if unknown_probabilities is not None:
        unknown_probabilities = np.asarray(unknown_probabilities, dtype=np.float64)
    if (not len(y_known)
            or (known_scores is not None and len(y_known) != len(known_scores))
            or known_probabilities.shape != (len(y_known), len(catalog["symbols"]))
            or len(unknown_scores) != len(unknown_kinds) or not len(unknown_scores)
            or (unknown_probabilities is not None and (
                unknown_probabilities.shape != (len(unknown_scores), len(catalog["symbols"]))
                or not np.isfinite(unknown_probabilities).all()
            ))
            or (known_scores is not None and not np.isfinite(known_scores).all())
            or not np.isfinite(unknown_scores).all()):
        raise ValueError("Los arrays de evaluación conocidos/desconocidos no están alineados.")

    if method == "softmax":
        if unknown_scores.shape != (len(unknown_kinds), 2):
            raise ValueError("Softmax desconocido requiere puntuación top-1 y margen.")
        known_top = known_probabilities.max(axis=1)
        ordered = np.sort(known_probabilities, axis=1)
        known_margin = known_top - ordered[:, -2]
        known_rejected = (known_top < 0.80) | (known_margin < 0.20)
        unknown_rejected = (unknown_scores[:, 0] < 0.80) | (unknown_scores[:, 1] < 0.20)
    elif method == "energy":
        if (known_scores is None or known_scores.shape != (len(y_known),)
                or unknown_scores.shape != (len(unknown_kinds),)
                or threshold is None or not np.isfinite(threshold)):
            raise ValueError("La estrategia de energía necesita un umbral finito calibrado.")
        known_rejected = known_scores > threshold
        unknown_rejected = unknown_scores > threshold
    else:
        raise ValueError("Estrategia de rechazo desconocida.")

    known_prediction = known_probabilities.argmax(axis=1)
    known_closed = classification_metrics(y_known, known_probabilities, catalog)
    accepted = ~known_rejected
    unknown_accepted = ~unknown_rejected
    per_class = []
    for class_index, symbol in enumerate(catalog["symbols"]):
        mask = y_known == class_index
        class_accepted = mask & accepted
        per_class.append({
            "id": symbol["id"],
            "count": int(mask.sum()),
            "false_rejection_rate": float(np.mean(known_rejected[mask])) if mask.any() else None,
            "closed_set_accuracy": float(np.mean(known_prediction[mask] == y_known[mask]))
            if mask.any() else None,
            "accepted_accuracy": (
                float(np.mean(known_prediction[class_accepted] == y_known[class_accepted]))
                if class_accepted.any() else None
            ),
        })
    per_kind = []
    for kind in sorted(set(unknown_kinds.tolist())):
        mask = unknown_kinds == kind
        per_kind.append({
            "kind": kind,
            "count": int(mask.sum()),
            "correct_rejection_rate": float(np.mean(unknown_rejected[mask])),
            "incorrect_acceptance_rate": float(np.mean(unknown_accepted[mask])),
        })
    return {
        "strategy": method,
        "energy_threshold": threshold if method == "energy" else None,
        "known_closed_set": known_closed,
        "known_false_rejection_rate": float(np.mean(known_rejected)),
        "known_accepted_count": int(accepted.sum()),
        "known_accepted_accuracy": (
            float(np.mean(known_prediction[accepted] == y_known[accepted]))
            if accepted.any() else None
        ),
        "known_by_class": per_class,
        "unknown_correct_rejection_rate": float(np.mean(unknown_rejected)),
        "unknown_incorrect_acceptance_rate": float(np.mean(unknown_accepted)),
        "unknown_count": int(len(unknown_scores)),
        "unknown_by_kind": per_kind,
        "confusion_known_with_reject": _known_confusion(
            y_known, known_prediction, known_rejected, len(catalog["symbols"])
        ),
        "confusion_unknown_with_reject": _unknown_confusion(
            unknown_probabilities, unknown_rejected, unknown_kinds,
            len(catalog["symbols"]),
        ),
        "unknown_confusion_rows": sorted(set(unknown_kinds.tolist())),
        "class_order": [symbol["id"] for symbol in catalog["symbols"]] + ["REJECT"],
    }


def _known_confusion(y, predicted, rejected, n_classes):
    matrix = np.zeros((n_classes, n_classes + 1), dtype=np.int64)
    columns = np.where(rejected, n_classes, predicted)
    np.add.at(matrix, (y, columns), 1)
    return matrix.tolist()


def _unknown_confusion(probabilities, rejected, kinds, n_classes):
    if probabilities is None:
        return None
    labels = sorted(set(kinds.tolist()))
    matrix = np.zeros((len(labels), n_classes + 1), dtype=np.int64)
    predicted = probabilities.argmax(axis=1)
    columns = np.where(rejected, n_classes, predicted)
    rows = {label: index for index, label in enumerate(labels)}
    for index, kind in enumerate(kinds):
        matrix[rows[kind], columns[index]] += 1
    return matrix.tolist()
