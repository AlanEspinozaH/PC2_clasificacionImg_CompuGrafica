"""Compare the current Softmax rejector with validation-calibrated energy rejection."""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from src.catalog import PROJECT_ROOT, load_catalog
from src.prediction import PredictionService
from src.rejection import (
    calibrate_energy_threshold,
    energy_scores,
    rejection_metrics,
)
from src.training_data import load_dataset
from src.unknown_data import (
    REQUIRED_EVALUATION_KINDS,
    UNKNOWN_KINDS,
    UnknownDrawingStore,
)
from src.preprocessing import preprocess_image


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _known_predictions(predictor, dataset, indices):
    logits = np.stack([predictor.logits(dataset.X[index]) for index in indices])
    return logits, _softmax(logits)


def _unknown_predictions(predictor, store, records, participants, image_size):
    selected = [row for row in records if row["participant_id"] in participants]
    if not selected:
        return [], np.empty((0, 6), dtype=np.float64)
    images = []
    for row in selected:
        if row["unknown_kind"] not in UNKNOWN_KINDS:
            raise ValueError(f"Etiqueta desconocida inválida en muestra {row['sample_id']}.")
        path = (store.root / row["path"]).resolve()
        if store.root.resolve() not in path.parents:
            raise ValueError("La ruta de una muestra desconocida sale de su directorio.")
        image_bytes = path.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != row["image_sha256"]:
            raise ValueError(f"El checksum no coincide para la muestra {row['sample_id']}.")
        with Image.open(path) as image:
            images.append(preprocess_image(image, size=image_size))
    logits = np.stack([predictor.logits(image) for image in images])
    return selected, logits


def _verify_partition(dataset, info, name):
    partition = info["partitions"][name]
    indices = np.asarray(partition["indices"], dtype=np.int64)
    if (indices.ndim != 1 or not len(indices) or (indices < 0).any()
            or (indices >= len(dataset.y)).any()):
        raise ValueError(f"La partición {name} del modelo no coincide con el dataset.")
    sample_ids = [dataset.records[index]["sample_id"] for index in indices]
    if sample_ids != partition["sample_ids"]:
        raise ValueError(f"Los sample_id de la partición {name} han cambiado.")
    return indices, set(partition["participants"])


def _macro_balanced_rejection_accuracy(metrics):
    class_acceptance = [
        1.0 - item["false_rejection_rate"]
        for item in metrics["known_by_class"] if item["count"]
    ]
    unknown_rejection = [
        item["correct_rejection_rate"]
        for item in metrics["unknown_by_kind"] if item["count"]
    ]
    return float((np.mean(class_acceptance) + np.mean(unknown_rejection)) / 2)


def _save_figures(output, catalog, test_report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [test_report["softmax"], test_report["energy"]]
    labels = ["Softmax actual", "Energía calibrada"]
    rates = [
        [item["known_false_rejection_rate"], item["unknown_incorrect_acceptance_rate"]]
        for item in metrics
    ]
    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    x = np.arange(2)
    width = 0.34
    axis.bar(x - width / 2, [row[0] for row in rates], width, label="Rechazo incorrecto · conocidos")
    axis.bar(x + width / 2, [row[1] for row in rates], width, label="Aceptación incorrecta · desconocidos")
    axis.set(xticks=x, xticklabels=labels, ylim=(0, 1), ylabel="Tasa", title="Evaluación final de rechazo")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output / "rejection_rates.png", dpi=160)
    plt.close(figure)

    matrix = np.asarray(test_report["energy"]["confusion_known_with_reject"])
    class_labels = [symbol["id"] for symbol in catalog["symbols"]] + ["REJECT"]
    figure, axis = plt.subplots(figsize=(7, 6), layout="constrained")
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    axis.set(
        xticks=range(len(class_labels)), yticks=range(len(class_labels)),
        xticklabels=class_labels, yticklabels=class_labels,
        xlabel="Predicción / rechazo", ylabel="Clase conocida real",
        title="Matriz de confusión conocida · energía",
    )
    axis.tick_params(axis="x", labelrotation=35)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Muestras")
    figure.savefig(output / "energy_confusion_matrix.png", dpi=160)
    plt.close(figure)

    matrix = np.asarray(test_report["energy"]["confusion_unknown_with_reject"])
    unknown_labels = ["UNKNOWN:" + kind for kind in test_report["energy"]["unknown_confusion_rows"]]
    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    image = axis.imshow(matrix, cmap="Oranges", vmin=0)
    axis.set(
        xticks=range(len(class_labels)), yticks=range(len(unknown_labels)),
        xticklabels=class_labels, yticklabels=unknown_labels,
        xlabel="Clase sugerida / rechazo", ylabel="Tipo real desconocido",
        title="Resultados OOD por tipo · energía",
    )
    axis.tick_params(axis="x", labelrotation=35)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Muestras")
    figure.savefig(output / "unknown_confusion_matrix.png", dpi=160)
    plt.close(figure)


def evaluate(model_root, data_root, unknown_root, output, write_policy=False):
    output = Path(output)
    if output.exists():
        raise ValueError(f"El directorio de resultados ya existe: {output}")
    policy_path = Path(model_root) / "rejection_calibration.json"
    if write_policy and policy_path.exists():
        raise ValueError(f"Ya existe una política de rechazo; no se sobrescribirá: {policy_path}")

    catalog = load_catalog()
    dataset = load_dataset(data_root)
    predictor = PredictionService(model_root, catalog)
    predictor.load()
    model_info = predictor.info
    if dataset.fingerprint != model_info.get("dataset_sha256"):
        raise ValueError("El dataset actual no coincide con el usado por este modelo.")

    validation_indices, validation_participants = _verify_partition(
        dataset, model_info, "validation"
    )
    test_indices, test_participants = _verify_partition(dataset, model_info, "test")
    if validation_participants & test_participants:
        raise ValueError("Validación y prueba comparten participantes.")

    unknown_store = UnknownDrawingStore(unknown_root, is_store_root=True)
    unknown_records = unknown_store.records()
    known_hashes = {record["image_sha256"] for record in dataset.records}
    if known_hashes & {record["image_sha256"] for record in unknown_records}:
        raise ValueError(
            "Una imagen aparece tanto en la colección conocida como en la desconocida."
        )
    training_participants = set(model_info["partitions"]["train"]["participants"])
    assigned_participants = training_participants | validation_participants | test_participants
    unknown_participants = {record["participant_id"] for record in unknown_records}
    if unknown_participants - assigned_participants:
        raise ValueError("Hay muestras desconocidas de participantes ajenos a este modelo.")
    unknown_validation, unknown_validation_logits = _unknown_predictions(
        predictor, unknown_store, unknown_records, validation_participants,
        catalog["image_size"],
    )
    unknown_test, unknown_test_logits = _unknown_predictions(
        predictor, unknown_store, unknown_records, test_participants,
        catalog["image_size"],
    )
    if not unknown_validation or not unknown_test:
        raise ValueError(
            "Se requieren dibujos desconocidos de participantes de validación y prueba "
            "distintos. Recolecta muestras antes de calibrar o evaluar."
        )
    for split_name, rows in (
        ("validación", unknown_validation),
        ("prueba", unknown_test),
    ):
        present = {row["unknown_kind"] for row in rows}
        missing = set(REQUIRED_EVALUATION_KINDS) - present
        if missing:
            names = ", ".join(UNKNOWN_KINDS[kind] for kind in sorted(missing))
            raise ValueError(
                f"Faltan tipos desconocidos en {split_name}: {names}. "
                "Recolecta cada tipo para los participantes reservados."
            )

    validation_logits, validation_probabilities = _known_predictions(
        predictor, dataset, validation_indices
    )
    test_logits, test_probabilities = _known_predictions(predictor, dataset, test_indices)
    energy_policy = calibrate_energy_threshold(
        validation_logits, unknown_validation_logits,
        dataset.y[validation_indices],
        [row["unknown_kind"] for row in unknown_validation],
    )
    test_known_energy = energy_scores(test_logits)
    test_unknown_energy = energy_scores(unknown_test_logits)
    validation_known_energy = energy_scores(validation_logits)
    validation_unknown_energy = energy_scores(unknown_validation_logits)
    validation_unknown_probabilities = _softmax(unknown_validation_logits)
    validation_unknown_softmax = np.column_stack((
        validation_unknown_probabilities.max(axis=1),
        np.sort(validation_unknown_probabilities, axis=1)[:, -1]
        - np.sort(validation_unknown_probabilities, axis=1)[:, -2],
    ))
    unknown_test_probabilities = _softmax(unknown_test_logits)
    test_unknown_softmax = np.column_stack((
        unknown_test_probabilities.max(axis=1),
        np.sort(unknown_test_probabilities, axis=1)[:, -1]
        - np.sort(unknown_test_probabilities, axis=1)[:, -2],
    ))
    validation_unknown_kinds = [row["unknown_kind"] for row in unknown_validation]
    test_unknown_kinds = [row["unknown_kind"] for row in unknown_test]

    validation_report = {
        "softmax": rejection_metrics(
            dataset.y[validation_indices], None,
            validation_probabilities, validation_unknown_softmax,
            validation_unknown_kinds, catalog, "softmax",
            unknown_probabilities=validation_unknown_probabilities,
        ),
        "energy": rejection_metrics(
            dataset.y[validation_indices], validation_known_energy,
            validation_probabilities, validation_unknown_energy,
            validation_unknown_kinds, catalog, "energy", energy_policy["threshold"],
            validation_unknown_probabilities,
        ),
    }
    test_report = {
        "softmax": rejection_metrics(
            dataset.y[test_indices], None, test_probabilities,
            test_unknown_softmax, test_unknown_kinds, catalog, "softmax",
            unknown_probabilities=unknown_test_probabilities,
        ),
        "energy": rejection_metrics(
            dataset.y[test_indices], test_known_energy, test_probabilities,
            test_unknown_energy, test_unknown_kinds, catalog, "energy",
            energy_policy["threshold"], unknown_test_probabilities,
        ),
    }
    softmax_validation_score = _macro_balanced_rejection_accuracy(
        validation_report["softmax"]
    )
    energy_validation_score = _macro_balanced_rejection_accuracy(
        validation_report["energy"]
    )
    energy_improves_validation = energy_validation_score > softmax_validation_score

    checkpoint_path = Path(model_root) / "model.pt"
    model_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": model_digest,
        "dataset_sha256": dataset.fingerprint,
        "validation_participants": sorted(validation_participants),
        "test_participants": sorted(test_participants),
        "known_validation_count": int(len(validation_indices)),
        "known_test_count": int(len(test_indices)),
        "unknown_validation_count": len(unknown_validation),
        "unknown_test_count": len(unknown_test),
        "unknown_training_ignored_count": sum(
            row["participant_id"] in training_participants for row in unknown_records
        ),
        "energy_calibration": energy_policy,
        "validation_macro_balanced_accuracy": {
            "softmax": softmax_validation_score,
            "energy": energy_validation_score,
            "energy_improves": energy_improves_validation,
        },
        "policy_written": bool(write_policy and energy_improves_validation),
        "validation": validation_report,
        "test": test_report,
        "warning": (
            "El rechazo es una estimación basada en estos ejemplos; no garantiza "
            "detectar todos los símbolos desconocidos."
        ),
    }

    output.mkdir(parents=True)
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "split", "sample_type", "sample_id", "participant_id", "truth", "top_1",
            "softmax_score", "softmax_margin", "energy",
            "softmax_rejected", "energy_rejected",
        ])
        for split_name, known_indices, probabilities, energies, rejected_energy in (
            ("validation", validation_indices,
             validation_probabilities, validation_known_energy,
             validation_known_energy > energy_policy["threshold"]),
            ("test", test_indices, test_probabilities, test_known_energy,
             test_known_energy > energy_policy["threshold"]),
        ):
            for row_index, dataset_index in enumerate(known_indices):
                sorted_probabilities = np.sort(probabilities[row_index])
                score = float(probabilities[row_index].max())
                margin = float(sorted_probabilities[-1] - sorted_probabilities[-2])
                writer.writerow([
                    split_name, "known", dataset.records[dataset_index]["sample_id"],
                    dataset.participants[dataset_index],
                    catalog["symbols"][int(dataset.y[dataset_index])]["id"],
                    catalog["symbols"][int(probabilities[row_index].argmax())]["id"],
                    score, margin, float(energies[row_index]),
                    score < 0.80 or margin < 0.20, bool(rejected_energy[row_index]),
                ])
        for split_name, unknown_rows, energies, probabilities in (
            ("validation", unknown_validation, validation_unknown_energy,
             validation_unknown_probabilities),
            ("test", unknown_test, test_unknown_energy, unknown_test_probabilities),
        ):
            for row_index, record in enumerate(unknown_rows):
                sorted_probabilities = np.sort(probabilities[row_index])
                score = float(probabilities[row_index].max())
                margin = float(sorted_probabilities[-1] - sorted_probabilities[-2])
                writer.writerow([
                    split_name, "unknown", record["sample_id"], record["participant_id"],
                    record["unknown_kind"],
                    catalog["symbols"][int(probabilities[row_index].argmax())]["id"],
                    score, margin, float(energies[row_index]),
                    score < 0.80 or margin < 0.20,
                    bool(energies[row_index] > energy_policy["threshold"]),
                ])
    _save_figures(output, catalog, test_report)

    if write_policy and energy_improves_validation:
        policy = {
            "schema_version": 1,
            "method": "energy",
            "energy_threshold": energy_policy["threshold"],
            "model_sha256": model_digest,
            "dataset_sha256": dataset.fingerprint,
            "validation_participants": sorted(validation_participants),
            "calibration": energy_policy,
            "created_at": report["created_at"],
        }
        with policy_path.open("x", encoding="utf-8") as stream:
            json.dump(policy, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument(
        "--unknown-data", type=Path, default=PROJECT_ROOT / "data" / "unknown"
    )
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "symbols")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "rejection-evaluation"
    )
    parser.add_argument(
        "--write-policy", action="store_true",
        help="Guarda una política calibrada nueva junto al modelo; nunca la sobrescribe.",
    )
    args = parser.parse_args()
    try:
        report = evaluate(
            args.model, args.data, args.unknown_data, args.output, args.write_policy
        )
        print(f"Resultados reales guardados en {args.output}")
        if args.write_policy and not report["policy_written"]:
            print("No se activó energía: no superó al rechazo Softmax en validación.")
        print(json.dumps(report["test"], indent=2, ensure_ascii=False))
        return 0
    except (ImportError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"No se pudo evaluar el rechazo: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
