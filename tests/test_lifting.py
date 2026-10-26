import unittest

import torch

from models.learnable_lifting import LearnableLifting2D


class LiftingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.model = LearnableLifting2D()

    def test_haar_reference_and_parameters(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 96)
        for name, kernel in self.model.effective_kernels().items():
            expected = torch.zeros(3, 1, 3, 3)
            expected[:, :, 1, 1] = 1.0 if name.startswith("P") else 0.5
            torch.testing.assert_close(kernel, expected, atol=0, rtol=0)
        x = torch.randn(2, 3, 224, 224)
        bands = self.model(x)
        self.assertEqual(bands.shape, (2, 12, 112, 112))
        a, b = x[:, :, 0::2], x[:, :, 1::2]
        row_low, row_high = (a + b) / 2, b - a
        ll = (row_low[:, :, :, 0::2] + row_low[:, :, :, 1::2]) / 2
        lh = row_low[:, :, :, 1::2] - row_low[:, :, :, 0::2]
        hl = (row_high[:, :, :, 0::2] + row_high[:, :, :, 1::2]) / 2
        hh = row_high[:, :, :, 1::2] - row_high[:, :, :, 0::2]
        torch.testing.assert_close(bands, torch.cat((ll, lh, hl, hh), 1))

    def test_inverse_after_learning(self):
        for p in self.model.parameters():
            torch.nn.init.normal_(p, std=0.05)
        x = torch.randn(2, 3, 224, 224)
        restored = self.model.inverse(self.model(x))
        self.assertLessEqual((restored - x).abs().max().item(), 1e-5)

    def test_constant_borders_and_sum_after_updates(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0)
        before = [p.detach().clone() for p in self.model.parameters()]
        for _ in range(3):
            optimizer.zero_grad()
            self.model(torch.randn(2, 3, 32, 32)).square().mean().backward()
            for p in self.model.parameters():
                self.assertTrue(p.requires_grad)
                self.assertIsNotNone(p.grad)
                self.assertGreater(p.grad.norm().item(), 0)
            optimizer.step()
            for name, kernel in self.model.effective_kernels().items():
                target = 1.0 if name.startswith("P") else 0.5
                torch.testing.assert_close(kernel.sum((1, 2, 3)), torch.full((3,), target), atol=1e-6, rtol=0)
        self.assertTrue(all(not torch.equal(a, b) for a, b in zip(before, self.model.parameters())))
        for name, kernel in self.model.effective_kernels().items():
            target = 1.0 if name.startswith("P") else 0.5
            torch.testing.assert_close(kernel.sum((1, 2, 3)), torch.full((3,), target), atol=1e-6, rtol=0)
        ll, lh, hl, hh = self.model.subbands(torch.ones(1, 3, 32, 32))
        torch.testing.assert_close(ll, torch.ones_like(ll), atol=1e-6, rtol=0)
        for band in (lh, hl, hh):
            self.assertLess(band.abs().max().item(), 1e-6)

    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 3, 31, 32))

    def test_no_channel_mixing_or_additional_parameters(self):
        self.assertEqual(set(self.model.state_dict()), {f"{name}.free" for name in ("P_f", "U_f", "P_c", "U_c")})
        for parameter in self.model.parameters():
            self.assertEqual(parameter.shape, (3, 8))
            torch.nn.init.normal_(parameter, std=0.02)
        x = torch.randn(1, 3, 32, 32)
        changed = x.clone()
        changed[:, 1] += torch.randn_like(changed[:, 1])
        before, after = self.model.subbands(x), self.model.subbands(changed)
        for a, b in zip(before, after):
            torch.testing.assert_close(a[:, [0, 2]], b[:, [0, 2]], atol=0, rtol=0)
