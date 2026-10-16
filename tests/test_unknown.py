"""Checks that unknown drawings remain separate from known-class training data."""

import base64
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from main import create_app
from src.catalog import load_catalog
from src.rejection import calibrate_energy_threshold, rejection_metrics


def drawing(offset=0):
    image = Image.new("RGB", (200, 200), "white")
    brush = ImageDraw.Draw(image)
    brush.line([(35 + offset, 35), (85 + offset, 130), (145, 45)], fill="black", width=8)
    return image


def data_url(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


class UnknownCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.app = create_app(self.root / "data", self.root / "missing-model")
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self):
        self.temporary.cleanup()

    def test_unknowns_are_separate_labeled_and_do_not_change_known_counts(self):
        response = self.client.post("/unknown", json={
            "image": data_url(drawing()),
            "unknown_kind": "integral",
            "participant_id": "p1",
            "input_method": "mouse",
        })
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json["participant_id"], "P01")
        self.assertEqual(response.json["counts"]["integral"], 1)
        self.assertEqual(sum(response.json["counts"].values()), 1)
        self.assertEqual(sum(self.client.get("/api/counts").json["counts"].values()), 0)
        self.assertTrue((self.root / "data" / "unknown" / "samples.csv").is_file())
        self.assertFalse((self.root / "data" / "raw" / "samples.csv").exists())

    def test_unknown_duplicates_and_invalid_labels_are_rejected(self):
        payload = {
            "image": data_url(drawing()),
            "unknown_kind": "sqrt",
            "participant_id": "P02",
        }
        self.assertEqual(self.client.post("/unknown", json=payload).status_code, 201)
        self.assertEqual(self.client.post("/unknown", json=payload).status_code, 409)
        payload["unknown_kind"] = "alpha"
        payload["image"] = data_url(drawing(offset=20))
        self.assertEqual(self.client.post("/unknown", json=payload).status_code, 400)

    def test_unknown_capture_requires_nonempty_valid_drawing_and_participant(self):
        image = Image.new("RGB", (200, 200), "white")
        payload = {"image": data_url(image), "unknown_kind": "random", "participant_id": "P03"}
        self.assertEqual(self.client.post("/unknown", json=payload).status_code, 400)
        payload["image"] = data_url(drawing())
        payload["participant_id"] = "P05"
        self.assertEqual(self.client.post("/unknown", json=payload).status_code, 400)


class RejectionEvaluationTests(unittest.TestCase):
    def test_energy_threshold_is_calibrated_from_validation_known_and_unknown_logits(self):
        known = np.array([[4.0, 0.0], [3.5, 0.0]])
        unknown = np.array([[0.1, 0.0], [0.0, 0.1]])
        calibration = calibrate_energy_threshold(
            known, unknown, known_groups=[0, 1],
            unknown_groups=["integral", "sqrt"],
        )
        self.assertEqual(calibration["method"], "energy")
        self.assertTrue(np.isfinite(calibration["threshold"]))
        self.assertEqual(calibration["validation_known_count"], 2)
        self.assertEqual(calibration["validation_unknown_count"], 2)
        self.assertEqual(calibration["validation_balanced_accuracy"], 1.0)
        with self.assertRaisesRegex(ValueError, "no están alineados"):
            calibrate_energy_threshold(known, unknown, known_groups=[0])

    def test_metrics_compare_softmax_with_energy_and_report_unknown_kinds(self):
        catalog = load_catalog()
        probabilities = np.zeros((2, 6), dtype=np.float64)
        probabilities[0, 0] = 0.99
        probabilities[1, 1] = 0.99
        unknown_softmax = np.array([[0.99, 0.98], [0.99, 0.98]])
        known_energy = np.array([-3.0, -3.0])
        unknown_energy = np.array([1.0, 1.2])
        softmax = rejection_metrics(
            np.array([0, 1]), np.zeros(2), probabilities, unknown_softmax,
            ["integral", "sqrt"], catalog, "softmax",
        )
        energy = rejection_metrics(
            np.array([0, 1]), known_energy, probabilities, unknown_energy,
            ["integral", "sqrt"], catalog, "energy", -1.0, probabilities,
        )
        self.assertEqual(softmax["unknown_incorrect_acceptance_rate"], 1.0)
        self.assertEqual(energy["unknown_correct_rejection_rate"], 1.0)
        self.assertEqual(len(energy["known_closed_set"]["confusion_matrix"]), 6)
        self.assertEqual({row["kind"] for row in energy["unknown_by_kind"]}, {"integral", "sqrt"})
        self.assertEqual(len(energy["confusion_known_with_reject"][0]), 7)
        self.assertEqual(energy["confusion_unknown_with_reject"], [[0, 0, 0, 0, 0, 0, 1],
                                                                    [0, 0, 0, 0, 0, 0, 1]])


if __name__ == "__main__":
    unittest.main()
