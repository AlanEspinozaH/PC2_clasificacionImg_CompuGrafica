"""Verifica separación por persona, métricas y la integración de la CNN."""

import base64
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from main import create_app
from src.catalog import load_catalog
from src.dataset import DatasetStore
from src.evaluation import classification_metrics
from src.training_data import PREPROCESSING, load_dataset, split_by_participant

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def make_collection(root, participants=("P01", "P02", "P03")):
    store = DatasetStore(root, load_catalog())
    for p, participant in enumerate(participants):
        for index, symbol in enumerate(store.catalog["symbols"]):
            image = Image.new("RGB", (200, 200), "white")
            brush = ImageDraw.Draw(image)
            brush.line([(30 + index * 5, 30), (60 + p * 5, 145), (140, 55 + index * 8)],
                       fill="black", width=7 + p)
            brush.ellipse((115, 110 + index * 2, 130 + p, 130 + index * 2), fill="black")
            store.save(image, symbol["id"], participant, "mouse")
    store.export()
    return root / "processed"


def png_payload():
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).ellipse((40, 40, 140, 130), outline="black", width=8)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return {"image": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()}


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prepared = make_collection(self.root / "data")

    def tearDown(self):
        self.temporary.cleanup()

    def test_participants_and_all_samples_are_separated(self):
        dataset = load_dataset(self.prepared)
        partitions = split_by_participant(dataset, seed=42)
        members = [set(dataset.participants[indices]) for indices in partitions.values()]
        self.assertFalse(members[0] & members[1])
        self.assertFalse(members[0] & members[2])
        self.assertFalse(members[1] & members[2])
        np.testing.assert_array_equal(np.sort(np.concatenate(list(partitions.values()))), np.arange(18))
        for indices in partitions.values():
            self.assertEqual(set(dataset.y[indices]), set(range(6)))
        repeated = split_by_participant(dataset, seed=42)
        for name in partitions:
            np.testing.assert_array_equal(partitions[name], repeated[name])

    def test_one_person_cannot_be_split_into_three_people(self):
        dataset = load_dataset(self.prepared)
        dataset.participants[:] = "P01"
        with self.assertRaisesRegex(ValueError, "3 personas"):
            split_by_participant(dataset)

    def test_explicit_partitions_reject_overlapping_people(self):
        dataset = load_dataset(self.prepared)
        with self.assertRaisesRegex(ValueError, "validación y prueba"):
            split_by_participant(dataset, validation_participants=["P02"], test_participants=["P02"])

    def test_missing_classes_in_a_partition_are_rejected(self):
        dataset = load_dataset(self.prepared)
        dataset.y[dataset.participants == "P03"] = 0
        with self.assertRaisesRegex(ValueError, "test no tiene"):
            split_by_participant(dataset, validation_participants=["P02"], test_participants=["P03"])

    def test_demo_samples_are_excluded_without_modifying_files(self):
        prepared = make_collection(self.root / "demo", participants=("P01", "P02", "DEMO01"))
        before = (prepared / "y.npy").read_bytes()
        dataset = load_dataset(prepared)
        self.assertEqual(dataset.excluded_demo, 6)
        self.assertEqual(len(dataset.y), 12)
        self.assertNotIn("DEMO01", dataset.participants)
        self.assertEqual(before, (prepared / "y.npy").read_bytes())

    def test_changed_catalog_is_rejected(self):
        path = self.prepared / "labels.json"
        catalog = json.loads(path.read_text())
        catalog["symbols"][0]["latex"] = "\\xi"
        path.write_text(json.dumps(catalog))
        with self.assertRaisesRegex(ValueError, "catálogo"):
            load_dataset(self.prepared)

    def test_arrays_cannot_silently_diverge_from_the_manifest(self):
        y = np.load(self.prepared / "y.npy", allow_pickle=False)
        y[0] = 1
        np.save(self.prepared / "y.npy", y, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "alineados"):
            load_dataset(self.prepared)

    def test_nonfinite_pixels_are_rejected(self):
        X = np.load(self.prepared / "X.npy", allow_pickle=False)
        X[0, 0, 0, 0] = np.nan
        np.save(self.prepared / "X.npy", X, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "píxeles inválidos"):
            load_dataset(self.prepared)

    def test_metrics_have_fixed_axes_and_a_top_three_baseline(self):
        probabilities = np.array([
            [0.9, 0.02, 0.02, 0.02, 0.02, 0.02],
            [0.03, 0.7, 0.15, 0.04, 0.04, 0.04],
            [0.01, 0.6, 0.2, 0.08, 0.06, 0.05],
        ])
        metrics = classification_metrics(np.array([0, 1, 2]), probabilities, load_catalog())
        self.assertAlmostEqual(metrics["top_1_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["top_3_accuracy"], 1)
        self.assertAlmostEqual(metrics["macro_f1"], 5 / 18)
        self.assertEqual(metrics["confusion_matrix"][2][1], 1)
        self.assertEqual(metrics["uniform_random_baseline"]["top_3_accuracy"], 0.5)

    def test_without_a_model_query_is_disabled_and_collection_still_works(self):
        client = create_app(self.root / "empty", self.root / "missing-model").test_client()
        self.assertFalse(client.get("/api/model").json["ready"])
        self.assertEqual(client.post("/predict", json=png_payload()).status_code, 503)
        payload = dict(png_payload(), class_id="alpha", participant_id="P04", input_method="mouse")
        self.assertEqual(client.post("/upload", json=payload).status_code, 201)


@unittest.skipUnless(TORCH_AVAILABLE, "Instala requirements-cpu.txt para verificar la CNN")
class ModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_root = self.root / "model"
        self.model_root.mkdir()
        self.catalog = load_catalog()

    def tearDown(self):
        self.temporary.cleanup()

    def write_known_model(self):
        import torch
        from src.model import SymbolCNN
        model = SymbolCNN()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.classifier[-1].bias.copy_(torch.arange(6, dtype=torch.float32))
        description = {"schema_version": 1, "architecture": SymbolCNN.architecture,
                       "catalog": self.catalog, "preprocessing": PREPROCESSING}
        torch.save(dict(description, state_dict=model.state_dict()), self.model_root / "model.pt")
        (self.model_root / "model_info.json").write_text(json.dumps(dict(description, purpose="pilot")))

    def test_top_three_commands_follow_the_trained_class_order_and_do_not_save_samples(self):
        self.write_known_model()
        client = create_app(self.root / "data", self.model_root).test_client()
        response = client.post("/predict", json=png_payload())
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["class_id"] for item in response.json["candidates"]], ["in", "partial", "equiv"])
        self.assertEqual([item["latex"] for item in response.json["candidates"]], ["\\in", "\\partial", "\\equiv"])
        self.assertTrue(response.json["pilot"])
        self.assertEqual(sum(client.get("/api/counts").json["counts"].values()), 0)

    def test_catalog_mismatch_blocks_prediction(self):
        self.catalog["symbols"][0]["latex"] = "\\xi"
        self.write_known_model()
        client = create_app(self.root / "data", self.model_root).test_client()
        self.assertFalse(client.get("/api/model").json["ready"])
        self.assertEqual(client.post("/predict", json=png_payload()).status_code, 503)

    def test_blank_drawing_is_rejected_before_prediction(self):
        self.write_known_model()
        client = create_app(self.root / "data", self.model_root).test_client()
        output = io.BytesIO()
        Image.new("RGB", (200, 200), "white").save(output, format="PNG")
        payload = {"image": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()}
        self.assertEqual(client.post("/predict", json=payload).status_code, 400)

    def test_training_artifact_can_be_loaded_by_the_application(self):
        from src.training import train_model
        dataset = load_dataset(make_collection(self.root / "collection"))
        output = self.root / "trained"
        info, metrics = train_model(dataset, output, epochs=2, batch_size=6, pilot=True,
                                    plots=False, threads=1)
        self.assertEqual(metrics["n_samples"], 6)
        self.assertTrue(metrics["preliminary"])
        self.assertTrue(1 <= info["best_epoch"] <= 2)
        self.assertEqual(len(info["partitions"]["train"]["participants"]), 1)
        with (output / "test_predictions.csv").open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 6)
        client = create_app(self.root / "query-data", output).test_client()
        self.assertTrue(client.get("/api/model").json["ready"])
        self.assertEqual(len(client.post("/predict", json=png_payload()).json["candidates"]), 3)

    def test_final_training_requires_the_collection_target(self):
        from src.training import train_model
        dataset = load_dataset(make_collection(self.root / "collection"))
        with self.assertRaisesRegex(ValueError, "meta por categoría"):
            train_model(dataset, self.root / "trained")
        self.assertFalse((self.root / "trained").exists())

    def test_external_data_stays_marked_as_a_reference_model(self):
        from src.training import train_model
        dataset = load_dataset(make_collection(self.root / "collection"))
        dataset.source_type = "hasyv2"
        dataset.source_description = "HASYv2 publico"
        output = self.root / "reference"
        info, metrics = train_model(dataset, output, epochs=1, pilot=True, plots=False, threads=1)
        self.assertEqual(info["purpose"], "external_baseline")
        self.assertTrue(metrics["preliminary"])
        client = create_app(self.root / "query-data", output).test_client()
        self.assertTrue(client.get("/api/model").json["pilot"])


if __name__ == "__main__":
    unittest.main()
