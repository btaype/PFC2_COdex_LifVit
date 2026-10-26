import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.metrics import binary_metrics, per_generator, select_validation_threshold
from training.checkpoints import load_checkpoint, save_checkpoint
from training.early_stopping import EarlyStopping
from training.scheduler import build_scheduler
from utils.config import load_config
from utils.reproducibility import restore_rng, rng_state, seed_everything


class TrainingTests(unittest.TestCase):
    def test_profiles_and_protocol(self):
        a, b = load_config(model="liftvit"), load_config(model="liftvit_h")
        self.assertEqual(a["training"]["batch_size"], 10)
        self.assertEqual(b["training"]["batch_size"], 4)
        self.assertEqual(a["vit"]["model_name"], "vit_base_patch16_224.augreg_in21k_ft_in1k")
        self.assertEqual(b["efficientnet"]["model_name"], "tf_efficientnet_b4.ns_jft_in1k")
        self.assertTrue(a["vit"]["pretrained"])
        self.assertTrue(b["efficientnet"]["pretrained"])
        self.assertFalse(a["data"]["require_group_ids"])
        ff = load_config("configs/ffpp.yaml", "liftvit_h")
        self.assertEqual(ff["protocol"], "ffpp_training")
        self.assertEqual(ff["training"]["batch_size"], 4)

    def test_early_stopping_and_small_best_improvements(self):
        stopper = EarlyStopping()
        self.assertFalse(stopper.update(0.8, 1))
        for epoch in range(2, 10):
            self.assertFalse(stopper.update(0.8001, epoch))
        self.assertTrue(stopper.update(0.8002, 10))

    def test_probabilities_auc_and_macro(self):
        y, p = [0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]
        metrics = binary_metrics(y, p)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(select_validation_threshold(y, p), 0.3)
        grouped = per_generator(y, p, ["real", "real", "ddpm", "ldm"], ["ddpm", "ldm"])
        self.assertEqual(grouped["ddpm"]["roc_auc"], 1.0)
        with self.assertRaises(ValueError):
            binary_metrics([0, 0], [0.1, 0.2])

    def test_checkpoint_resume_matches_uninterrupted_optimizer(self):
        seed_everything(42)
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        scheduler = build_scheduler(optimizer, 10)
        def step(p, opt, sched):
            opt.zero_grad()
            (p * torch.rand(1)).square().sum().backward()
            opt.step()
            sched.step()
        step(parameter, optimizer, scheduler)
        state = {"epoch": 1, "model_name": "liftvit", "model_state_dict": {"weight": parameter.detach().clone()},
                 "training_config": {}, "preprocessing_config": {}, "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "rng": rng_state()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pth"
            save_checkpoint(path, state)
            loaded = load_checkpoint(path, "liftvit")
            step(parameter, optimizer, scheduler)
            resumed = torch.nn.Parameter(loaded["model_state_dict"]["weight"].clone())
            opt = torch.optim.AdamW([resumed], lr=0.01)
            sched = build_scheduler(opt, 10)
            opt.load_state_dict(loaded["optimizer"])
            sched.load_state_dict(loaded["scheduler"])
            restore_rng(loaded["rng"])
            step(resumed, opt, sched)
            torch.testing.assert_close(parameter, resumed, atol=0, rtol=0)
            self.assertEqual(scheduler.get_last_lr(), sched.get_last_lr())
            with self.assertRaises(ValueError):
                load_checkpoint(path, "liftvit_h")

    def test_partial_accumulation_matches_mean(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        p = torch.tensor(2.0, requires_grad=True)
        for item in x:
            ((p * item).square() / len(x)).backward()
        q = torch.tensor(2.0, requires_grad=True)
        (q * x).square().mean().backward()
        torch.testing.assert_close(p.grad, q.grad)
