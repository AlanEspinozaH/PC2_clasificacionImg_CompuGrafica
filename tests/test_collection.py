"""Comprueba que la captura preserve etiquetas, participantes y trazos."""

import base64
import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from main import create_app
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
        self.assertTrue((self.data_root / "raw" / record["path"]).exists())

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
