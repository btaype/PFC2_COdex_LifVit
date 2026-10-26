import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.weights import load_backbone, public_model_kwargs


class WeightsTests(unittest.TestCase):
    def test_public_cache_uses_state_dict_loader_and_preserves_exact_file(self):
        with patch("models.weights.cached_public_checkpoint", return_value=Path("cache/model.safetensors")):
            args = public_model_kwargs("vit_base_patch16_224.augreg_in21k_ft_in1k", Path("cache"))
        self.assertEqual(args["pretrained_cfg_overlay"]["file"], str(Path("cache/model.safetensors")))
        self.assertFalse(args["pretrained_cfg_overlay"]["custom_load"])
        with patch("models.weights.cached_public_checkpoint", return_value=None):
            self.assertNotIn("pretrained_cfg_overlay", public_model_kwargs("vit_base_patch16_224.augreg_in21k_ft_in1k", Path("cache")))

    def test_backbone_loading_rejects_missing_and_mismatched(self):
        module = torch.nn.LayerNorm(3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pth"
            torch.save({"state_dict": {"module.weight": torch.ones(3), "module.bias": torch.ones(3)}}, path)
            report = load_backbone(module, path, ("head.",))
            self.assertEqual(len(report["loaded"]), 2)
            torch.testing.assert_close(module.bias, torch.ones(3))
            torch.save({"weight": torch.ones(4), "bias": torch.ones(3)}, path)
            with self.assertRaisesRegex(ValueError, "dimensiones"):
                load_backbone(module, path, ("head.",))
            torch.save({"weight": torch.ones(3)}, path)
            with self.assertRaisesRegex(ValueError, "Faltan"):
                load_backbone(module, path, ("head.",))
