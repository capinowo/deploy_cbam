"""
Definisi class custom CBAM untuk loading checkpoint YOLOv12n (+CBAM).

Checkpoint .pt nyimpen objek module ter-pickle (atribut: pool, fc, act), jadi
forward-nya harus pakai atribut itu, BUKAN self.mlp / avg_pool / max_pool.

SafeChannelAttention.forward di sini backward-compatible:
  - versi lama : fc = Conv2d biasa, Sigmoid dipanggil terpisah lewat self.act
  - versi baru : fc = Sequential yang sudah termasuk Sigmoid di dalamnya
Jadi checkpoint 20e / 35e / 50e dan finetuned semuanya bisa jalan.
"""

import torch
import torch.nn as nn


def _fc_has_sigmoid(fc) -> bool:
    """True kalau Sigmoid sudah ada di dalam fc (versi baru)."""
    if isinstance(fc, nn.Sigmoid):
        return True
    if isinstance(fc, nn.Sequential):
        return any(isinstance(m, nn.Sigmoid) for m in fc.modules())
    return False


class SafeChannelAttention(nn.Module):
    def __init__(self, channels: int = 16, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        # kalau Sigmoid belum ada di fc, pakai self.act (checkpoint versi lama)
        if not _fc_has_sigmoid(self.fc):
            act = getattr(self, "act", None)
            w = act(w) if act is not None else torch.sigmoid(w)
        return x * w


# dipertahankan supaya checkpoint yang nyimpen referensi ke ChannelAttention tetap ke-load
class ChannelAttention(SafeChannelAttention):
    pass


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out)) * x


class CBAM(nn.Module):
    def __init__(self, channels: int = 16, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = SafeChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


def register_custom_classes_in_main():
    """
    Biar torch.load bisa nemuin class custom ini walaupun checkpoint
    di-pickle dari notebook Kaggle (module '__main__').
    """
    import sys

    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for cls in (ChannelAttention, SafeChannelAttention, SpatialAttention, CBAM):
            setattr(main_module, cls.__name__, cls)

    try:
        torch.serialization.add_safe_globals(
            [ChannelAttention, SafeChannelAttention, SpatialAttention, CBAM]
        )
    except AttributeError:
        pass