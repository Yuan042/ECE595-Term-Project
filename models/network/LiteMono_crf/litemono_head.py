# litemono_head.py
import torch
import torch.nn as nn


class DSConvBlock(nn.Module):
    """LiteMono風格：Depthwise(3x3)->BN->ReLU->Pointwise(1x1)->BN->ReLU + 殘差"""

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1,
                            groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.dw(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.pw(out)
        out = self.bn2(out)
        out = out + identity
        return self.act(out)


class LiteMonoCRF(nn.Module):
    """
    介面相容 NewCRF：
      __init__(input_dim, embed_dim, v_dim, window_size=..., num_heads=..., depth=3, concat_v=True, **kwargs)
      forward(x, v) -> Tensor[B, embed_dim, H, W]
    保留 v (cost-volume) 作為額外資訊；輸出 embed_dim 供 PixelShuffle(2) 使用（embed_dim 必須是4的倍數）。
    """

    def __init__(self,
                 input_dim: int = 96,
                 embed_dim: int = 96,
                 v_dim:     int = 64,
                 window_size: int = 7,   # 佔位，為相容保留參數名
                 num_heads:   int = 4,   # 佔位
                 depth:       int = 3,   # 疊幾個DSConvBlock
                 concat_v:    bool = True,
                 norm:        str = "bn",
                 **kwargs):
        super().__init__()
        self.concat_v = concat_v
        in_ch = input_dim + (v_dim if concat_v else 0)

        norm2d = nn.BatchNorm2d if norm == "bn" else lambda c: nn.GroupNorm(
            8, c)

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, 1, bias=False),
            norm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        blocks = [DSConvBlock(embed_dim) for _ in range(depth)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, v):
        if self.concat_v:
            x = torch.cat([x, v], dim=1)
        y = self.proj(x)
        y = self.blocks(y)
        return y
