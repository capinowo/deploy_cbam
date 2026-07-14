"""
PENTING - BACA DULU SEBELUM DEPLOY:

File ini isinya definisi class custom (ChannelAttention, SpatialAttention, CBAM,
SafeChannelAttention) yang HARUS SAMA PERSIS dengan yang dipakai waktu training
di Kaggle. Checkpoint .pt hasil training nyimpen referensi ke class-class ini di
dalam pickle-nya, jadi kalau nama class atau isinya beda, nanti muncul error kayak:

    AttributeError: Can't get attribute 'SafeChannelAttention' on <module '__main__'>

Aku (Claude) nulis implementasi CBAM standar di bawah ini sebagai starting point,
TAPI kamu WAJIB cek ulang ke notebook training kamu di Kaggle dan samain persis:
  - nama class
  - urutan & isi layer di __init__
  - forward pass

Cara cek cepat: buka notebook Kaggle training, cari definisi class
SafeChannelAttention / CBAM kamu, copy-paste ke sini gantiin implementasi di bawah.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


# alias -- GANTI isinya kalau versi training kamu beda dari ChannelAttention biasa
class SafeChannelAttention(ChannelAttention):
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
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = SafeChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


def register_custom_classes_in_main():
    """
    Trik supaya torch.load bisa nemuin class custom ini walaupun checkpoint
    di-pickle dari notebook Kaggle (module '__main__'), bukan dari file .py ini.
    Dipanggil sebelum load model di model_loader.py.
    """
    import sys

    main_module = sys.modules.get("__main__")
    if main_module is None:
        return

    for cls in (ChannelAttention, SafeChannelAttention, SpatialAttention, CBAM):
        setattr(main_module, cls.__name__, cls)

    # daftar juga ke torch safe globals (dibutuhkan torch >= 2.6 kalau weights_only=True)
    try:
        torch.serialization.add_safe_globals(
            [ChannelAttention, SafeChannelAttention, SpatialAttention, CBAM]
        )
    except AttributeError:
        pass  # versi torch lama belum punya add_safe_globals, aman diabaikan