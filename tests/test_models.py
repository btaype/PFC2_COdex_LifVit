import unittest

import torch

from models import build_model
from scripts.smoke_test import synthetic_step
from training.optimizer import build_optimizer, set_training_stage
from utils.config import load_config


class ModelTests(unittest.TestCase):
    def test_liftvit_forward_backward_update(self):
        torch.set_num_threads(2)
        result = synthetic_step("liftvit", torch.device("cpu"))
        self.assertEqual(result["intermediate_shapes"]["frequency_tokens"], [1, 197, 768])

    def test_liftvit_h_forward_backward_update(self):
        torch.set_num_threads(2)
        result = synthetic_step("liftvit_h", torch.device("cpu"))
        self.assertEqual(result["intermediate_shapes"]["spatial_tokens"], [1, 49, 768])

    def test_progressive_freezing_preserves_lifting_gradient(self):
        torch.set_num_threads(2)
        cfg = load_config(model="liftvit_h")
        model = build_model(cfg, initialize=False).train()
        optimizer = build_optimizer(model, cfg)
        included = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(included, {id(p) for p in model.parameters()})
        lifting_ids = {id(p) for p in model.frequency.lifting.parameters()}
        lifting_groups = [g for g in optimizer.param_groups if g["name"].startswith("lifting/")]
        self.assertEqual({id(p) for g in lifting_groups for p in g["params"]}, lifting_ids)
        self.assertTrue(all(g["weight_decay"] == 0 and g["lr"] == cfg["training"]["lr_lifting"] for g in lifting_groups))
        set_training_stage(model, 1)
        self.assertFalse(any(p.requires_grad for p in model.frequency.vit.parameters()))
        model(torch.rand(1, 3, 224, 224)).sum().backward()
        self.assertTrue(all(p.grad is not None and p.grad.norm() > 0 for p in model.frequency.lifting.parameters()))
        set_training_stage(model, 3)
        self.assertTrue(all(p.requires_grad for p in model.frequency.lifting.parameters()))
        self.assertFalse(next(model.frequency.vit.blocks[0].parameters()).requires_grad)
        self.assertTrue(next(model.frequency.vit.blocks[-1].parameters()).requires_grad)
        set_training_stage(model, 6)
        self.assertTrue(all(p.requires_grad for p in model.frequency.lifting.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.frequency.vit.parameters()))
        self.assertFalse(next(model.spatial.backbone.blocks[0].parameters()).requires_grad)
        self.assertTrue(next(model.spatial.backbone.blocks[-1].parameters()).requires_grad)
        for layer in model.spatial.backbone.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                self.assertFalse(layer.training)
