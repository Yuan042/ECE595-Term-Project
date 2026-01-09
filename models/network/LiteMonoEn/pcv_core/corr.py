import torch
import torch.nn as nn
import torch.nn.functional as F
from models.network.LiteMonoEn.pcv_core.utils import bilinear_sampler
import math

try:
    import corr_sampler
except:
    corr_sampler = None

try:
    import alt_cuda_corr
except:
    alt_cuda_corr = None


class CorrBlock1D(nn.Module):
    """
    1D correlation pyramid for stereo along disparity (width) axis.
    - sample_num:    number of 1D samples per location (odd number recommended)
    - num_levels:    pyramid levels (>=1)
    - downsample:    stride to build the next level before compression
    - compress_factor: width compression factor per level (>=1, 1 means no compression)
    """

    def __init__(self, fmap1, fmap2, sample_num=9, num_levels=4,
                 downsample=2, compress_factor: int = 1):
        super().__init__()
        assert sample_num > 0 and sample_num % 2 == 1, "sample_num 建議為奇數"
        self.sample_num = int(sample_num)
        self.num_levels = int(max(1, num_levels))
        self.downsample = int(max(1, downsample))
        self.compress_factor = int(max(1, compress_factor))

        # base correlation [B,H,W1,1,W2] -> reshape to [B*H*W1, 1, 1, W2]
        corr = CorrBlock1D.corr(fmap1, fmap2)                   # (B,H,W1,1,W2)
        B, H, W1, _, W2 = corr.shape
        corr = corr.reshape(B * H * W1, 1, 1, W2)               # (BHW,1,1,W2)

        # index for 1D sampling（改用 arange）
        half = self.sample_num // 2
        self.register_buffer("index", torch.arange(-half, half + 1, dtype=torch.float32).view(1, 1, 1, self.sample_num),
                             persistent=False)

        # build pyramid with width-safe compression
        self.corr_pyramid = nn.ModuleList()
        corr_levels = [corr]  # level-0
        cur = corr
        for lvl in range(1, self.num_levels):
            # 先做寬度方向的安全壓縮
            W = cur.shape[-1]
            cf = max(1, min(self.compress_factor, W))
            if cf > 1 and W >= cf:
                cur = F.avg_pool2d(cur, (1, cf), stride=(1, cf))
            # 存進列表
            corr_levels.append(cur)

        # register 一下（保持與 nn.Module 一致）
        for t in corr_levels:
            self.corr_pyramid.append(nn.Identity())  # 佔位
        # 用 buffers 存實際 tensor
        self._corr_levels = corr_levels

    def forward(self, coords, sigma, test_mode=False):
        """
        coords: (B, G, H, W1)  disparity means for each Gaussian component
        sigma:  (B, G, H, W1)  disparity std (per Gaussian)
        return: (B, C, H, W1)  concatenated correlations over levels & samples
        """
        B, G, H, W1 = coords.shape
        # (B,G,H,W1) -> (BHW, 1, G, 1)
        sigma_ = sigma.permute(0, 2, 3, 1).contiguous().reshape(
            B * H * W1, 1, G, 1)
        coords_ = coords.permute(
            0, 2, 3, 1).contiguous().reshape(B * H * W1, 1, G, 1)

        # broadcasting: (1,1,1,S)
        dx = self.index.to(coords_.device)
        # x: (BHW,1,G,S)
        x = dx * sigma_ + coords_

        out_pyramid = []
        for i in range(self.num_levels):
            corr_i = self._corr_levels[i]                 # (BHW,1,1,W_i)
            # 等比例縮放 sample 位置（配合前面的壓縮層數）
            denom = (self.compress_factor ** i)
            if denom > 1:
                x0 = x / float(denom)
            else:
                x0 = x

            # (BHW,1, G*S, 1)
            x0 = x0.reshape(B * H * W1, 1, G * self.sample_num, 1)
            y0 = torch.zeros_like(x0)
            coords_lvl = torch.cat([x0, y0], dim=-1)     # (BHW,1,G*S,2)

            # 抽樣 (BHW,1, G*S, 1) -> (BHW,1, G*S, 1)
            corr_samp = bilinear_sampler(
                corr_i.contiguous(), coords_lvl.contiguous())
            # reshape 回 (B,H,W1, G*S)
            corr_samp = corr_samp.view(B, H, W1, -1)
            out_pyramid.append(corr_samp)

        out = torch.cat(out_pyramid, dim=-1)             # (B,H,W1, Csum)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        """
        fmap1: (B, C, H, W1)
        fmap2: (B, C, H, W2)
        return: (B, H, W1, 1, W2)
        """
        B, C, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        # einsum over channels -> (B,H,W1,W2)
        corr = torch.einsum('bchw, bchW -> bhwW', fmap1, fmap2)
        # normalize by sqrt(C)
        corr = corr / math.sqrt(float(C))
        return corr.unsqueeze(3).contiguous()            # (B,H,W1,1,W2)
