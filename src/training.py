"""Entrenamiento en CPU, selección con validación y una evaluación final de prueba."""

import copy
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluation import classification_metrics
from .model import SymbolCNN
from .training_data import PREPROCESSING, split_by_participant


def evaluate(model, loader):
    model.eval()
    probabilities, labels = [], []
    loss_total = 0.0
    with torch.inference_mode():
        for images, y in loader:
            logits = model(images)
            loss_total += float(nn.functional.cross_entropy(logits, y, reduction="sum"))
            probabilities.append(logits.softmax(dim=1).numpy())
            labels.append(y.numpy())
    y = np.concatenate(labels)
    return loss_total / len(y), y, np.concatenate(probabilities)


def save_plots(history, metrics, catalog, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Entrenamiento")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Validación")
    axes[0].set(xlabel="Época", ylabel="Entropía cruzada", title="Pérdida")
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Entrenamiento")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="Validación")
    axes[1].set(xlabel="Época", ylabel="Accuracy top-1", ylim=(0, 1), title="Exactitud")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)

    matrix = np.asarray(metrics["confusion_matrix"])
    labels = [s["id"] for s in catalog["symbols"]]
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    image = ax.imshow(matrix, cmap="Greens", vmin=0)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
           yticklabels=labels, xlabel="Predicción", ylabel="Clase real", title="Matriz de confusión · prueba")
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center",
                    color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=ax, label="Muestras")
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def train_model(dataset, output, epochs=25, batch_size=32, patience=5, seed=42,
                pilot=False, validation_participants=None, test_participants=None,
                plots=True, threads=4):
    output = Path(output)
    if output.exists():
        raise ValueError("El directorio del modelo ya existe. Usa otro --output para conservar el experimento.")
    if min(epochs, batch_size, patience, threads) < 1:
        raise ValueError("Épocas, batch size, paciencia e hilos deben ser positivos.")
    counts = np.bincount(dataset.y, minlength=len(dataset.catalog["symbols"]))
    if not pilot and (counts < dataset.catalog["target_per_class"]).any():
        raise ValueError(
            "Falta completar la meta por categoría. Para un ensayo preliminar usa --pilot; "
            "sus métricas no son una evaluación definitiva."
        )
    partitions = split_by_participant(dataset, seed, validation_participants, test_participants)
    torch.set_num_threads(min(threads, os.cpu_count() or 1))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    images = torch.from_numpy(np.ascontiguousarray(dataset.X.transpose(0, 3, 1, 2)))
    labels = torch.from_numpy(dataset.y.copy())

    def loader(name, shuffle=False):
        indices = partitions[name]
        return DataLoader(TensorDataset(images[indices], labels[indices]), batch_size=batch_size,
                          shuffle=shuffle, generator=torch.Generator().manual_seed(seed))

    train_loader, validation_loader = loader("train", True), loader("validation")
    model = SymbolCNN(len(dataset.catalog["symbols"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_loss, best_epoch, best_state, waiting = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total, correct, total = 0.0, 0, 0
        for batch, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
        validation_loss, y_validation, probabilities = evaluate(model, validation_loader)
        if not np.isfinite(loss_total) or not np.isfinite(validation_loss):
            raise ValueError("El entrenamiento produjo una pérdida no finita.")
        row = {"epoch": epoch, "train_loss": loss_total / total, "train_accuracy": correct / total,
               "validation_loss": validation_loss,
               "validation_accuracy": float(np.mean(probabilities.argmax(1) == y_validation))}
        history.append(row)
        print(f"Época {epoch:02d}: pérdida={row['train_loss']:.4f}; "
              f"validación={validation_loss:.4f}; acc_val={row['validation_accuracy']:.3f}", flush=True)
        if validation_loss < best_loss - 0.00001:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                break
    model.load_state_dict(best_state)
    test_loss, y_test, probabilities = evaluate(model, loader("test"))
    metrics = classification_metrics(y_test, probabilities, dataset.catalog)
    metrics.update(loss=test_loss, preliminary=pilot)
    split_info = {
        name: {"indices": indices.tolist(),
               "sample_ids": [dataset.records[i]["sample_id"] for i in indices],
               "participants": sorted(set(dataset.participants[indices].tolist())),
               "n_samples": len(indices),
               "class_counts": np.bincount(dataset.y[indices], minlength=len(counts)).tolist()}
        for name, indices in partitions.items()
    }
    info = {
        "schema_version": 1, "architecture": SymbolCNN.architecture,
        "created_at": datetime.now(timezone.utc).isoformat(), "purpose": "pilot" if pilot else "collected_dataset",
        "catalog": dataset.catalog, "preprocessing": PREPROCESSING,
        "dataset_sha256": dataset.fingerprint, "n_samples": len(dataset.y),
        "excluded_demo": dataset.excluded_demo, "class_counts": counts.tolist(),
        "seed": seed, "best_epoch": best_epoch, "epochs_run": len(history),
        "selection": "minimum validation cross entropy; test evaluated once after selection",
        "augmentation": "none", "device": "cpu", "torch_version": str(torch.__version__),
        "batch_size": batch_size, "patience": patience, "partitions": split_info,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="training-", dir=output.parent) as temporary:
        staging = Path(temporary) / "model"
        staging.mkdir()
        torch.save({"schema_version": 1, "architecture": SymbolCNN.architecture,
                    "catalog": dataset.catalog, "preprocessing": PREPROCESSING,
                    "state_dict": best_state}, staging / "model.pt")
        for filename, value in [("model_info.json", info), ("metrics.json", metrics),
                                ("history.json", history), ("split.json", split_info)]:
            (staging / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        with (staging / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_id", "participant_id", "real", "top_1", "top_2", "top_3"])
            for index, ranking in zip(partitions["test"], np.argsort(-probabilities, axis=1, kind="stable")):
                writer.writerow([dataset.records[index]["sample_id"], dataset.participants[index],
                                 dataset.catalog["symbols"][dataset.y[index]]["id"],
                                 *[dataset.catalog["symbols"][i]["id"] for i in ranking[:3]]])
        if plots:
            save_plots(history, metrics, dataset.catalog, staging)
        staging.rename(output)
    return info, metrics
