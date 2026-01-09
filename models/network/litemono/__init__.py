import torch
import torch.nn as nn

from models.network.litemono.depth_encoder import LiteMono as LiteMonoEncoder
from models.network.litemono.depth_decoder import DepthDecoder


class LiteMonoDepth(nn.Module):
    """
    Wrap Lite-Mono encoder+decoder and output depth (meters).
    """

    def __init__(self, in_chans=3, height=192, width=640,
                 min_depth=0.1, max_depth=100.0, **kwargs):
        super().__init__()
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        # mapping constants for inverse depth
        self.beta = 1.0 / self.max_depth
        self.alpha = 1.0 / self.min_depth - 1.0 / self.max_depth

        # Lite-Mono encoder (paper default model='lite-mono')
        self.encoder = LiteMonoEncoder(in_chans=in_chans,
                                       model='lite-mono',
                                       height=height, width=width)

        # Lite-Mono decoder expects encoder.num_ch_enc
        self.decoder = DepthDecoder(self.encoder.num_ch_enc,
                                    # 0..2 (1/4,1/8,1/16 in Lite-Mono)
                                    scales=range(3),
                                    num_output_channels=1,
                                    use_skips=True)

    def _disp_to_depth(self, disp):
        # disp: [B,1,H,W] in [0,1] after sigmoid in decoder
        # inverse depth in [1/d_max, 1/d_min]
        inv_d = self.alpha * disp + self.beta
        # depth in [d_min, d_max]
        depth = 1.0 / torch.clamp(inv_d, min=1e-6)
        return depth

    def forward(self, x):
        # list of 3 feature maps (1/4, 1/8, 1/16)
        feats = self.encoder(x)
        outs = self.decoder(feats)            # dict {("disp", s): [B,1,H,W]}
        # full-res disparity (upsampled inside decoder)
        disp = outs[("disp", 0)]
        depth = self._disp_to_depth(disp)
        return depth
