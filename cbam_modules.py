import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import CBAM, ChannelAttention, SpatialAttention


class SafeChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()  # `reduction` diabaikan, sama seperti di notebook training
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.act = nn.Sigmoid()

    def _conv(self) -> nn.Conv2d:
        fc = self._modules.get("fc")
        if isinstance(fc, nn.Conv2d):
            return fc
        convs = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        if len(convs) == 1:
            return convs[0]
        raise AttributeError(
            f"SafeChannelAttention: butuh tepat 1 layer Conv2d, ditemukan {len(convs)} di checkpoint ini."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self._conv()(F.adaptive_avg_pool2d(x, 1)))


def register_custom_classes_in_main():
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return

    classes = (SafeChannelAttention, ChannelAttention, SpatialAttention, CBAM)
    for cls in classes:
        setattr(main_module, cls.__name__, cls)

    try:
        torch.serialization.add_safe_globals(list(classes))
    except AttributeError:
        pass