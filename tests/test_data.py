import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data.sampler import BalancedBatchSampler
from data.splits import build_manifest, group_audit, load_records, read_manifest, validate_records


class DataTests(unittest.TestCase):
    def test_sampler_balances_without_losing_examples(self):
        labels = [0] * 13 + [1] * 7
        sampler = BalancedBatchSampler(labels, 10)
        batches = list(sampler)
        seen = set()
        for batch in batches:
            self.assertEqual(len(batch), 10)
            self.assertEqual(sum(labels[i] for i in batch), 5)
            seen.update(batch)
        self.assertEqual(seen, set(range(len(labels))))
        self.assertEqual(batches, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(batches, list(sampler))

    def test_reject_group_leak_and_duplicates(self):
        a = {"path": "a.png", "split": "train", "label": 0, "generator": "real", "group_id": "person", "sha256": "a", "pixel_sha256": "pa"}
        b = dict(a, path="b.png", split="test", sha256="b", pixel_sha256="pb")
        with self.assertRaisesRegex(ValueError, "Fuga"):
            validate_records([a, b])
        with self.assertRaisesRegex(ValueError, "duplicada"):
            validate_records([a, dict(b, group_id="other", pixel_sha256="pa")])

    def test_manifest_integrity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train" / "real").mkdir(parents=True)
            image = root / "train" / "real" / "a.png"
            Image.fromarray(np.full((32, 32, 3), 123, dtype=np.uint8)).save(image)
            meta = root / "meta.csv"
            meta.write_text("path,group_id\ntrain/real/a.png,person_1\n", encoding="utf-8")
            manifest = root / "manifest.csv"
            build_manifest(root, manifest, meta)
            self.assertEqual(len(read_manifest(manifest, root)), 1)
            with self.assertRaises(FileExistsError):
                build_manifest(root, manifest, meta)
            Image.fromarray(np.full((32, 32, 3), 124, dtype=np.uint8)).save(image)
            with self.assertRaisesRegex(ValueError, "modificado"):
                read_manifest(manifest, root)

    def test_folders_work_without_invented_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "train" / "real" / "a.png"
            image.parent.mkdir(parents=True)
            Image.fromarray(np.full((32, 32, 3), 42, dtype=np.uint8)).save(image)
            cfg = {"root": str(root), "manifest": str(root / "manifest.csv"), "auto_manifest": True, "require_group_ids": False}
            records = load_records(cfg, create_if_missing=True)
            self.assertEqual(records[0]["group_id"], "")
            self.assertEqual(records[0]["source_id"], "")
            self.assertEqual(group_audit(records)["group_check_coverage"], "unavailable")
            self.assertEqual(records, load_records(cfg))
            with self.assertRaisesRegex(ValueError, "Falta group_id"):
                read_manifest(root / "manifest.csv", root, require_groups=True)
            Image.fromarray(np.full((32, 32, 3), 43, dtype=np.uint8)).save(root / "train" / "real" / "b.png")
            with self.assertRaisesRegex(ValueError, "carpetas cambiaron"):
                load_records(cfg, create_if_missing=True)

    def test_optional_metadata_still_rejects_known_group_leaks(self):
        a = {"path": "a.png", "split": "train", "label": 0, "generator": "real", "group_id": "same_person", "sha256": "a", "pixel_sha256": "pa"}
        b = dict(a, path="b.png", split="test", sha256="b", pixel_sha256="pb")
        unknown = dict(a, path="c.png", group_id="", sha256="c", pixel_sha256="pc")
        with self.assertRaisesRegex(ValueError, "Fuga"):
            validate_records([a, b, unknown], require_groups=False)
        validate_records([a, unknown], require_groups=False)

    def test_empty_folders_do_not_create_a_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            with self.assertRaisesRegex(ValueError, "No hay imágenes"):
                load_records({"root": str(root), "manifest": str(manifest)}, create_if_missing=True)
            self.assertFalse(manifest.exists())
