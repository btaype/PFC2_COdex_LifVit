import torch
from torch import nn
from torch.nn import functional as F


class _ConstrainedDepthwise(nn.Module):
    def __init__(self, total):
        super().__init__()
        self.free = nn.Parameter(torch.zeros(3, 8))
        self.total = total

    def kernel(self):
        center = self.total - self.free.sum(dim=1, keepdim=True)
        return torch.cat((self.free[:, :4], center, self.free[:, 4:]), dim=1).reshape(3, 1, 3, 3)

    def forward(self, x):
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), self.kernel(), groups=3)


class LearnableLifting2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.P_f = _ConstrainedDepthwise(1.0)
        self.U_f = _ConstrainedDepthwise(0.5)
        self.P_c = _ConstrainedDepthwise(1.0)
        self.U_c = _ConstrainedDepthwise(0.5)

    def _columns(self, x):
        even, odd = x[:, :, :, 0::2], x[:, :, :, 1::2]
        high = odd - self.P_c(even)
        return even + self.U_c(high), high

    def subbands(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[2] % 2 or x.shape[3] % 2:
            raise ValueError("LLS requiere [B,3,H,W] con H/W pares")
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            even, odd = x[:, :, 0::2, :], x[:, :, 1::2, :]
            high = odd - self.P_f(even)
            low = even + self.U_f(high)
            ll, lh = self._columns(low)
            hl, hh = self._columns(high)
            return ll, lh, hl, hh

    def forward(self, x):
        return torch.cat(self.subbands(x), dim=1)

    def _inverse_columns(self, low, high):
        even = low - self.U_c(high)
        odd = high + self.P_c(even)
        return torch.stack((even, odd), dim=-1).flatten(-2)

    def inverse(self, bands):
        ll, lh, hl, hh = bands.chunk(4, dim=1) if isinstance(bands, torch.Tensor) else bands
        with torch.autocast(device_type=ll.device.type, enabled=False):
            low = self._inverse_columns(ll.float(), lh.float())
            high = self._inverse_columns(hl.float(), hh.float())
            even = low - self.U_f(high)
            odd = high + self.P_f(even)
            return torch.stack((even, odd), dim=3).flatten(2, 3)

    def effective_kernels(self):
        return {name: getattr(self, name).kernel() for name in ("P_f", "U_f", "P_c", "U_c")}
