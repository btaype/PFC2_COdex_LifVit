from torch import nn


class BidirectionalCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.frequency_attention = nn.MultiheadAttention(768, 12, dropout=0.1, batch_first=True)
        self.spatial_attention = nn.MultiheadAttention(768, 12, dropout=0.1, batch_first=True)
        self.frequency_norm1 = nn.LayerNorm(768)
        self.spatial_norm1 = nn.LayerNorm(768)
        self.frequency_norm2 = nn.LayerNorm(768)
        self.spatial_norm2 = nn.LayerNorm(768)
        self.frequency_ffn = self._ffn()
        self.spatial_ffn = self._ffn()
        self.dropout = nn.Dropout(0.1)

    @staticmethod
    def _ffn():
        return nn.Sequential(nn.Linear(768, 3072), nn.GELU(), nn.Dropout(0.1), nn.Linear(3072, 768), nn.Dropout(0.1))

    def forward(self, frequency, spatial, return_attention=False):
        f, fw = self.frequency_attention(frequency, spatial, spatial, need_weights=return_attention, average_attn_weights=False)
        s, sw = self.spatial_attention(spatial, frequency, frequency, need_weights=return_attention, average_attn_weights=False)
        f = self.frequency_norm1(frequency + self.dropout(f))
        s = self.spatial_norm1(spatial + self.dropout(s))
        f = self.frequency_norm2(f + self.frequency_ffn(f))
        s = self.spatial_norm2(s + self.spatial_ffn(s))
        return (f, s, fw, sw) if return_attention else (f, s)
