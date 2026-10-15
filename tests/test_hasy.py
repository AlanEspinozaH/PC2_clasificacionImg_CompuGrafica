"""Comprueba procedencia, etiquetas, duplicados y aislamiento de HASYv2."""

import csv
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.catalog import load_catalog
from src.hasy_import import file_digest, image_key, prepare_hasy
from src.training_data import load_dataset, split_by_participant, split_dataset


def fixture_archive(path, wrong_label=False, duplicate=False, missing_image=False):
    catalog = load_catalog()
    symbols = [{"symbol_id": s["hasy_symbol_id"], "latex": s["latex"]} for s in catalog["symbols"]]
    if wrong_label:
        symbols[0]["latex"] = "\\xi"
    rows, images = [], []
    first_png = None
    for p in range(3):
        for index, symbol in enumerate(catalog["symbols"]):
            name = f"hasy-data/v2-{p * 6 + index:05d}.png"
            image = Image.new("RGB", (32, 32), "white")
            ImageDraw.Draw(image).line([(3 + index, 3), (15 + p, 24), (26, 5 + index)], fill="black", width=2 + p)
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            encoded = stream.getvalue()
            if p == 0 and index == 0:
                first_png = encoded
            if duplicate and p == 1 and index == 0:
                encoded = first_png
            rows.append({"path": name, "symbol_id": symbol["hasy_symbol_id"],
                         "latex": symbol["latex"], "user_id": p + 1})
            images.append(("HASYv2/" + name, encoded))

    def csv_bytes(items):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)
        return stream.getvalue().encode()

    files = [("HASYv2/symbols.csv", csv_bytes(symbols)),
             ("HASYv2/hasy-data-labels.csv", csv_bytes(rows)), *images]
    if missing_image:
        files.pop()
    with tarfile.open(path, "w:bz2") as tar:
        for name, content in files:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return file_digest(path)


class HasyImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive = self.root / "test.tar.bz2"
        self.destination = self.root / "data" / "hasy"

    def tearDown(self):
        self.temporary.cleanup()

    def test_import_preserves_source_ids_labels_and_the_own_collection(self):
        digest = fixture_archive(self.archive)
        own = self.root / "data" / "raw" / "samples.csv"
        own.parent.mkdir(parents=True)
        own.write_text("coleccion propia")
        metadata = prepare_hasy(self.archive, self.destination, per_class=2, expected_md5=digest)
        dataset = load_dataset(self.destination / "processed")
        self.assertEqual(metadata["n_samples"], 12)
        self.assertFalse(metadata["complete_collection"])
        self.assertFalse(metadata["participant_identity_reliable"])
        self.assertEqual(dataset.source_type, "hasyv2")
        self.assertEqual(own.read_text(), "coleccion propia")
        np.testing.assert_array_equal(np.bincount(dataset.y, minlength=6), [2] * 6)
        with (self.destination / "processed" / "source_manifest.csv").open(newline="") as stream:
            provenance = list(csv.DictReader(stream))
        self.assertEqual(len(provenance), 12)
        for record in provenance:
            self.assertEqual(record["participant_id"], "HASY_" + record["source_user_id"])
            symbol = dataset.catalog["symbols"][int(record["class_index"])]
            self.assertEqual(record["source_symbol_id"], str(symbol["hasy_symbol_id"]))
            self.assertTrue((self.destination / "raw" / record["path"]).is_file())

    def test_checksum_mismatch_does_not_create_a_dataset(self):
        fixture_archive(self.archive)
        with self.assertRaisesRegex(ValueError, "checksum"):
            prepare_hasy(self.archive, self.destination, per_class=2, expected_md5="invalid")
        self.assertFalse(self.destination.exists())

    def test_wrong_id_to_latex_mapping_is_rejected(self):
        digest = fixture_archive(self.archive, wrong_label=True)
        with self.assertRaisesRegex(ValueError, "no corresponde"):
            prepare_hasy(self.archive, self.destination, per_class=2, expected_md5=digest)
        self.assertFalse(self.destination.exists())

    def test_duplicate_drawings_are_removed_before_sampling(self):
        digest = fixture_archive(self.archive, duplicate=True)
        metadata = prepare_hasy(self.archive, self.destination, per_class=2, expected_md5=digest)
        self.assertEqual(metadata["skipped"]["duplicates"], 1)
        dataset = load_dataset(self.destination / "processed")
        self.assertEqual(len({r["image_sha256"] for r in dataset.records}), 12)

    def test_missing_images_do_not_leave_a_partial_dataset(self):
        digest = fixture_archive(self.archive, missing_image=True)
        with self.assertRaisesRegex(ValueError, "todas las imágenes"):
            prepare_hasy(self.archive, self.destination, per_class=2, expected_md5=digest)
        self.assertFalse(self.destination.exists())

    def test_existing_dataset_is_preserved(self):
        digest = fixture_archive(self.archive)
        self.destination.mkdir(parents=True)
        marker = self.destination / "keep.txt"
        marker.write_text("conservar")
        with self.assertRaisesRegex(ValueError, "ya existe"):
            prepare_hasy(self.archive, self.destination, per_class=2, expected_md5=digest)
        self.assertEqual(marker.read_text(), "conservar")

    def test_paths_cannot_escape_the_archive_namespace(self):
        for value in ("../hasy-data/v2-001.png", "/hasy-data/v2-001.png", "other/file.png"):
            with self.subTest(path=value), self.assertRaises(ValueError):
                image_key(value)

    def test_shared_hasy_user_id_does_not_prevent_image_stratification(self):
        digest = fixture_archive(self.archive)
        prepare_hasy(self.archive, self.destination, per_class=3, expected_md5=digest)
        dataset = load_dataset(self.destination / "processed")
        dataset.participants[:] = "HASY_16925"
        partitions = split_dataset(dataset, seed=42)
        np.testing.assert_array_equal(np.sort(np.concatenate(list(partitions.values()))), np.arange(18))
        for name, indices in partitions.items():
            np.testing.assert_array_equal(np.bincount(dataset.y[indices], minlength=6), [1] * 6)
            np.testing.assert_array_equal(indices, split_dataset(dataset, seed=42)[name])
        with self.assertRaisesRegex(ValueError, "no identifican personas"):
            split_by_participant(dataset)

    def test_hasy_does_not_claim_person_holdouts(self):
        digest = fixture_archive(self.archive)
        prepare_hasy(self.archive, self.destination, per_class=3, expected_md5=digest)
        dataset = load_dataset(self.destination / "processed")
        with self.assertRaisesRegex(ValueError, "no permite reservar personas"):
            split_dataset(dataset, validation_participants=["HASY_1"], test_participants=["HASY_2"])

    def test_too_few_hasy_images_cannot_form_three_partitions(self):
        digest = fixture_archive(self.archive)
        prepare_hasy(self.archive, self.destination, per_class=2, expected_md5=digest)
        with self.assertRaisesRegex(ValueError, "tres imágenes por clase"):
            split_dataset(load_dataset(self.destination / "processed"))


if __name__ == "__main__":
    unittest.main()
