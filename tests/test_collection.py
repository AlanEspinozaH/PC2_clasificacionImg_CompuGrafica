"""Comprueba que la captura preserve etiquetas, participantes y trazos."""

import base64
import csv
import io
import json
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from main import create_app
from src.catalog import load_catalog
from src.dataset import DatasetStore, FIELDS
from src.preprocessing import EmptyDrawingError, preprocess_image


def drawing(offset=0, transparent=False):
    image = Image.new(
        "RGBA" if transparent else "RGB", (200, 200),
        (0, 0, 0, 0) if transparent else (255, 255, 255),
    )
    brush = ImageDraw.Draw(image)
    brush.line([(40 + offset, 40), (90 + offset, 120)], fill="black", width=8)
    brush.line([(90 + offset, 120), (130 + offset, 65)], fill="black", width=8)
    return image


def data_url(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


class PreprocessingTests(unittest.TestCase):
    def test_transparency_does_not_erase_the_drawing(self):
        opaque = preprocess_image(drawing())
        transparent = preprocess_image(drawing(transparent=True))
        np.testing.assert_array_equal(opaque, transparent)
        self.assertEqual(opaque.shape, (32, 32, 1))
        self.assertEqual(opaque.dtype, np.float32)
        self.assertEqual(float(opaque.min()), 0.0)
        self.assertGreater(float(opaque.max()), 0.9)

    def test_translation_is_removed_by_centering(self):
        np.testing.assert_array_equal(
            preprocess_image(drawing()), preprocess_image(drawing(offset=25))
        )

    def test_aspect_ratio_is_preserved(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((50, 30, 69, 109), fill="black")
        result = preprocess_image(image)[..., 0]
        y_coordinates, x_coordinates = np.where(result > 0.5)
        width = x_coordinates.max() - x_coordinates.min() + 1
        height = y_coordinates.max() - y_coordinates.min() + 1
        self.assertLess(abs(width / height - 20 / 80), 0.08)

    def test_blank_images_are_rejected_with_both_backgrounds(self):
        for image in (
            Image.new("RGB", (200, 200), "white"),
            Image.new("RGBA", (200, 200), (0, 0, 0, 0)),
        ):
            with self.subTest(mode=image.mode):
                with self.assertRaises(EmptyDrawingError):
                    preprocess_image(image)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temporary.name)
        self.app = create_app(self.data_root)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self):
        self.temporary.cleanup()

    def upload(self, class_id="alpha", participant_id="P01", image=None):
        return self.client.post("/upload", json={
            "class_id": class_id,
            "participant_id": participant_id,
            "image": data_url(image if image is not None else drawing()),
            "input_method": "mouse",
        })

    def records(self):
        with (self.data_root / "raw" / "samples.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            return list(csv.DictReader(stream))

    def test_capture_records_the_selected_class_and_participant(self):
        response = self.upload("partial", "p02")
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json["counts"]["partial"], 1)
        self.assertEqual(response.json["participant_id"], "P02")
        with (self.data_root / "raw" / "samples.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            record = next(csv.DictReader(stream))
        self.assertEqual(record["class_id"], "partial")
        self.assertEqual(record["class_index"], "4")
        self.assertEqual(record["participant_id"], "P02")
        self.assertRegex(
            record["path"], rf"^partial/P02_partial_0001_{record['sample_id'][:8]}\.png$"
        )
        self.assertRegex(record["sample_id"], r"^[0-9a-f]{32}$")
        self.assertRegex(record["image_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(list(record), FIELDS)
        self.assertTrue((self.data_root / "raw" / record["path"]).exists())

    def test_participant_variants_are_normalized_to_the_assigned_code(self):
        for offset, value in enumerate(("p1", "P1", " P01 ")):
            with self.subTest(value=value):
                response = self.upload(participant_id=value, image=drawing(offset * 5))
                self.assertEqual(response.status_code, 201)
                self.assertEqual(response.json["participant_id"], "P01")
        self.assertEqual([row["participant_id"] for row in self.records()], ["P01"] * 3)

    def test_correlatives_are_independent_by_participant_and_category(self):
        cases = [
            ("alpha", "P01", 0, "P01_alpha_0001_"),
            ("alpha", "P01", 5, "P01_alpha_0002_"),
            ("xi", "P01", 10, "P01_xi_0001_"),
            ("alpha", "P02", 15, "P02_alpha_0001_"),
        ]
        for class_id, participant_id, offset, _prefix in cases:
            self.assertEqual(
                self.upload(class_id, participant_id, drawing(offset)).status_code, 201
            )
        for row, (_class_id, _participant_id, _offset, prefix) in zip(
            self.records(), cases
        ):
            self.assertTrue(Path(row["path"]).name.startswith(prefix))

    def test_correlative_survives_application_restart_and_counts_legacy_names(self):
        self.assertEqual(self.upload(image=drawing()).status_code, 201)
        record = self.records()[0]
        original = self.data_root / "raw" / record["path"]
        legacy_relative = Path("alpha") / f"{record['sample_id']}.png"
        legacy = self.data_root / "raw" / legacy_relative
        original.rename(legacy)
        record["path"] = legacy_relative.as_posix()
        record["participant_id"] = "P1"  # Variante que admitía el recolector anterior.
        with (self.data_root / "raw" / "samples.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(record)
        legacy_bytes = legacy.read_bytes()

        restarted = create_app(self.data_root)
        restarted.config["TESTING"] = True
        self.client = restarted.test_client()
        self.assertEqual(self.upload(image=drawing(20)).status_code, 201)

        records = self.records()
        self.assertEqual(records[0]["path"], legacy_relative.as_posix())
        self.assertEqual(records[0]["participant_id"], "P1")
        self.assertEqual(records[1]["participant_id"], "P01")
        self.assertTrue(Path(records[1]["path"]).name.startswith("P01_alpha_0002_"))
        self.assertEqual(legacy.read_bytes(), legacy_bytes)
        self.assertEqual(self.client.post("/prepare").status_code, 200)

    def test_existing_destination_is_never_overwritten(self):
        store = DatasetStore(self.data_root, load_catalog())
        category = self.data_root / "raw" / "alpha"
        category.mkdir(parents=True)
        first_id = uuid.UUID(hex="a" * 32)
        second_id = uuid.UUID(hex="b" * 32)
        collision = category / "P01_alpha_0001_aaaaaaaa.png"
        collision.write_bytes(b"archivo existente")
        with (
            patch.object(store, "_next_correlative", return_value=1),
            patch("src.dataset.uuid.uuid4", side_effect=[first_id, second_id]),
        ):
            record = store.save(drawing(), "alpha", "P01", "mouse")
        self.assertEqual(collision.read_bytes(), b"archivo existente")
        self.assertEqual(Path(record["path"]).name, "P01_alpha_0001_bbbbbbbb.png")
        self.assertEqual(record["sample_id"], "b" * 32)

    def test_identical_images_do_not_inflate_the_collection(self):
        self.assertEqual(self.upload().status_code, 201)
        self.assertEqual(self.upload().status_code, 409)
        counts = self.client.get("/api/counts").json["counts"]
        self.assertEqual(sum(counts.values()), 1)
        self.assertEqual(len(list((self.data_root / "raw").glob("*/*.png"))), 1)

    def test_unknown_class_and_invalid_participant_do_not_save_files(self):
        self.assertEqual(self.upload("../outside").status_code, 400)
        self.assertEqual(self.upload(participant_id="").status_code, 400)
        self.assertEqual(self.upload(participant_id="../P01").status_code, 400)
        self.assertEqual(self.upload(participant_id="P05").status_code, 400)
        self.assertEqual(self.upload(participant_id="DEMO01").status_code, 400)
        self.assertFalse((self.data_root / "raw").exists())

    def test_blank_capture_is_rejected(self):
        response = self.upload(image=Image.new("RGB", (200, 200), "white"))
        self.assertEqual(response.status_code, 400)
        self.assertFalse((self.data_root / "raw").exists())

    def test_invalid_png_is_rejected(self):
        response = self.client.post("/upload", json={
            "class_id": "alpha", "participant_id": "P01",
            "image": "data:image/png;base64," + base64.b64encode(b"invalid").decode("ascii"),
        })
        self.assertEqual(response.status_code, 400)

    def test_non_json_request_is_rejected(self):
        self.assertEqual(self.client.post("/upload", data="not-json").status_code, 400)

    def test_export_has_aligned_labels_participants_and_images(self):
        self.assertEqual(self.upload("xi", "P01", drawing()).status_code, 201)
        second_image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(second_image).ellipse((50, 50, 150, 120), outline="black", width=8)
        self.assertEqual(self.upload("partial", "P02", second_image).status_code, 201)
        response = self.client.post("/prepare")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["metadata"]["X_shape"], [2, 32, 32, 1])
        self.assertFalse(response.json["metadata"]["complete_collection"])
        downloaded = self.client.get("/dataset.zip")
        self.assertEqual(downloaded.status_code, 200)
        archive_bytes = downloaded.data
        downloaded.close()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            X = np.load(io.BytesIO(archive.read("X.npy")), allow_pickle=False)
            y = np.load(io.BytesIO(archive.read("y.npy")), allow_pickle=False)
            participants = np.load(
                io.BytesIO(archive.read("participants.npy")), allow_pickle=False
            )
            labels = json.loads(archive.read("labels.json"))
        self.assertEqual(X.shape, (2, 32, 32, 1))
        np.testing.assert_array_equal(y, [1, 4])
        np.testing.assert_array_equal(participants, ["P01", "P02"])
        self.assertEqual(labels["symbols"][1]["id"], "xi")
        self.assertEqual(labels["symbols"][4]["id"], "partial")
        self.assertGreaterEqual(float(X.min()), 0.0)
        self.assertLessEqual(float(X.max()), 1.0)

    def test_empty_collection_cannot_be_exported(self):
        self.assertEqual(self.client.post("/prepare").status_code, 400)
        self.assertEqual(self.client.get("/dataset.zip").status_code, 404)


if __name__ == "__main__":
    unittest.main()
