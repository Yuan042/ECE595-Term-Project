# models/network/litemono/encoder_wrapper.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registry import MODELS
from .depth_encoder import LiteMono  # 你的檔案路徑
# ↑ 這個 LiteMono 就是你貼進來的那個類別。 :contentReference[oaicite:5]{index=5}


@MODELS.register_module(name='LiteMonoEncoder')
class LiteMonoEncoderWrapper(nn.Module):
    """
    輸出 dict: {'1/4': f4, '1/8': f8, '1/16': f16, '1/32': f32}
    並把每個尺度的通道用 1x1 conv 投到 decoder 需要的 in_channels。
    """

    def __init__(self, opt):
        super().__init__()
        # 從 YAML 讀（例：lite-mono / lite-mono-tiny / lite-mono-8m）
        lm_cfg = getattr(opt.model, 'litemono', {})
        model_name = lm_cfg.get('model', 'lite-mono')
        in_chans = lm_cfg.get('in_chans', 3)

        self.backbone = LiteMono(in_chans=in_chans, model=model_name)

        # decoder.in_channels 例如 [64,128,320,512]（你自己在 YAML 設）
        out_chs = list(getattr(opt.model.decoder, 'in_channels'))
        # LiteMono 3 個 stage 的原始通道（看 depth_encoder.py 的 num_ch_enc）
        # e.g. base: [48,80,128]；tiny: [32,64,128]；8m: [64,128,224]  :contentReference[oaicite:6]{index=6}
        in_chs_3 = list(self.backbone.num_ch_enc)
        in_chs_4 = in_chs_3 + [in_chs_3[-1]]  # 第4個尺度先沿用最後一層

        self.proj = nn.ModuleList([
            nn.Conv2d(ci, co, kernel_size=1, bias=False)
            for ci, co in zip(in_chs_4, out_chs)
        ])

    def forward(self, x):
        # list: [f4, f8, f16]  （LiteMono forward 回傳 features） :contentReference[oaicite:7]{index=7}
        feats = self.backbone(x)
        f4, f8, f16 = feats
        f32 = F.avg_pool2d(f16, 2, 2)  # 補出 1/32

        f4 = self.proj[0](f4)
        f8 = self.proj[1](f8)
        f16 = self.proj[2](f16)
        f32 = self.proj[3](f32)
        return {'1/4': f4, '1/8': f8, '1/16': f16, '1/32': f32}
