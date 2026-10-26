import unittest

import torch

from models.cross_attention import BidirectionalCrossAttention


class AttentionTests(unittest.TestCase):
    def test_both_directions_use_original_inputs(self):
        torch.set_num_threads(2)
        module = BidirectionalCrossAttention().eval()
        frequency = torch.randn(1, 197, 768, requires_grad=True)
        spatial = torch.randn(1, 49, 768, requires_grad=True)
        f, s, fw, sw = module(frequency, spatial, return_attention=True)
        self.assertEqual(f.shape, frequency.shape)
        self.assertEqual(s.shape, spatial.shape)
        self.assertEqual(fw.shape, (1, 12, 197, 49))
        self.assertEqual(sw.shape, (1, 12, 49, 197))
        direct = module.spatial_attention(spatial, frequency, frequency, need_weights=False)[0]
        expected = module.spatial_norm1(spatial + direct)
        expected = module.spatial_norm2(expected + module.spatial_ffn(expected))
        torch.testing.assert_close(s, expected, atol=1e-5, rtol=1e-4)
        (f[:, :, 0].square().mean() + s[:, :, 1].square().mean()).backward()
        self.assertGreater(frequency.grad.norm().item(), 0)
        self.assertGreater(spatial.grad.norm().item(), 0)
