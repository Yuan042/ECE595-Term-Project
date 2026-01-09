# models/network/litemono/litemono.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from .third_party.litemono.networks.depth_encoder import LiteMono as DepthNetEncoder


class LiteMonoEnBackbone(nn.Module):
    """輸出四層：[f4, f8, f16, f32]，並提供 ppm_in_channels。"""

    def __init__(self, in_chans=3, model='tiny'):
        super().__init__()

        # 允許在 YAML/程式用簡短名；映射到第三方真正接受的字串
        _model = str(model).lower()
        alias = {
            "tiny":  "lite-mono-tiny",
            "small": "lite-mono-small",
            "base":  "lite-mono",        # 論文原版
            "8m":    "lite-mono-8m",
            # 也容忍已是完整名的情形
            "lite-mono-tiny":  "lite-mono-tiny",
            "lite-mono-small": "lite-mono-small",
            "lite-mono":       "lite-mono",
            "lite-mono-8m":    "lite-mono-8m",
        }
        lm_model = alias.get(_model, "lite-mono")  # 預設用 base

        # 建立第三方編碼器（注意：這裡的 DepthNetEncoder 是 LiteMono 的別名）
        self.encoder = DepthNetEncoder(
            in_chans=max(in_chans, 1), model=lm_model)

        # ---- 安全檢查：確保第三方已設定 dims ----
        if not hasattr(self.encoder, "dims"):
            # 代表傳入的 model 名稱仍未命中第三方邏輯；印出提示並直接拋錯，避免後面更難追
            raise RuntimeError(
                f"LiteMono encoder 未填入 dims（model='{model}' → 傳入第三方 '{lm_model}' 仍失敗）。"
                "請確認 third_party/litemono/networks/depth_encoder.py 的 __init__ 中，"
                "對應這個 model 名稱有設定 self.dims。"
            )

        self._ppm_in_channels = None

    def forward(self, x: torch.Tensor):
        """
        輸入: x (B,C,H,W)
        輸出: [f4, f8, f16, f32]  # 依 SupDepth 要求
        """
        feats = self.encoder(x)  # 可能是 list/tuple/dict，且可能只有3層

        # ---- 標準化型別 ----
        if isinstance(feats, (list, tuple)):
            feats = list(feats)
        elif isinstance(feats, dict):
            try:
                feats = [feats[k]
                         for k in sorted(feats.keys(), key=lambda k: int(str(k)))]
            except Exception:
                feats = list(feats.values())
        else:
            raise RuntimeError(
                f"encoder.forward 應回 list/tuple/dict，但拿到 {type(feats)}")

        # ---- 對齊為四尺度 ----
        if len(feats) == 3:
            f4, f8, f16 = feats
            # 方案1：平均池化做 f32（簡單穩定）
            f32 = F.avg_pool2d(f16, kernel_size=2, stride=2)
            # 方案2：學習型下採樣，開啟則把上面那行改成 f32 = self.ds32(f16)
            # if not hasattr(self, 'ds32'):
            #     self.ds32 = nn.Conv2d(f16.size(1), f16.size(1), kernel_size=3, stride=2, padding=1, bias=False).to(f16.device)
            # f32 = self.ds32(f16)
        elif len(feats) >= 4:
            f4, f8, f16, f32 = feats[:4]
        else:
            raise RuntimeError(
                f"encoder 只回 {len(feats)} 個尺度，無法對齊到 [f4,f8,f16,f32]")

        # ---- 第一次 forward 才確定通道數（lazy init）----
        if self._ppm_in_channels is None:
            self._ppm_in_channels = [
                f4.size(1), f8.size(1), f16.size(1), f32.size(1)]
            # 若後續層需要通道（例如 PSP/head），可在這裡建立/重建對應層

        return [f4, f8, f16, f32]

    @property
    def ppm_in_channels(self):
        # 若尚未 forward，回退到 encoder 的 dims（可能不存在）
        return self._ppm_in_channels
