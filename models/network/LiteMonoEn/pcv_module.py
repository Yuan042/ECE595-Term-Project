# models/network/LiteMonoEn/pcv_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.network.LiteMonoEn.pcv_core.corr import CorrBlock1D
from models.network.LiteMonoEn.pcv_core.update import BasicMultiUpdateBlock
from models.network.LiteMonoEn.pcv_core.utils import coords_grid
from types import SimpleNamespace


class PCVAdapter(nn.Module):
    def __init__(
        self,
        K=3,
        proj_out=32,
        n_downsample=2,
        corr_levels=4,
        sample_num=9,
        hidden_dims=(128, 128, 128),
        compress_factor=1,
        init_mu=0.0,
        init_sigma=1.0,
        ctx04_channels=544, ctx08_channels=384, ctx16_channels=256
    ):
        super().__init__()
        self.K = int(K)
        self.proj_out = int(proj_out)
        self.n_downsample = int(n_downsample)
        self.corr_levels = int(corr_levels)
        self.sample_num = int(sample_num)
        self.hidden_dims = tuple(hidden_dims)
        self.compress_factor = int(compress_factor)
        self.init_mu = init_mu
        self.init_sigma = init_sigma
        self.args = SimpleNamespace(
            n_gru_layers=3,
            corr_levels=self.corr_levels,
            sample_num=self.sample_num,
            n_downsample=self.n_downsample,
            mixed_precision=False,
            slow_fast_gru=False,
            valid_iters=12,
            gauss_num=self.K,
            hidden_dims=self.hidden_dims,
            # 注意：init_mu 需是長度 = gauss_num 的 tuple
            init_mu=tuple([float(self.init_mu)] * self.K),
            init_sigma=float(self.init_sigma),
        )

        self.updater = BasicMultiUpdateBlock(
            args=self.args,
            hidden_dims=self.hidden_dims,      # (128, 128, 128)
            encoder_output_dim_04=128,
            encoder_output_dim_08=128,
            encoder_output_dim_16=128,
        )

        # ---- 保存到 self（後續 forward / initialize / CorrBlock 都會用到）----

        # 供 CRF 接收的 PCV 輸出通道投影（[w, mu, sigma] → 3K channels）
        self.proj = nn.Sequential(
            nn.Conv2d(3 * self.K, self.proj_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.proj_out),
            nn.ReLU(inplace=True),
        )

        # 1/4 尺度特徵通道對齊（依你 PPM 輸出的 C 調整；現用 512→128）
        self.f4_proj = nn.Conv2d(
            in_channels=512, out_channels=128, kernel_size=1, bias=False)

        # ---- 構建 PCV 的 Update Block 參數（只保留一份 _Args）----
        # ---- keep other imports and code above ----
        # === Debug: log & assert auto-derived dims (PCVAdapter) ===
        def _as_tuple(x):
            try:
                return tuple(int(t) for t in x)
            except Exception:
                try:
                    return int(x)
                except Exception:
                    return str(x)

        self._arch_sig = {
            "cls": self.__class__.__name__,
            "hidden_dims": _as_tuple(getattr(self, "hidden_dims", None)),
            "corr_levels": _as_tuple(getattr(self, "corr_levels", None)),
            "c_pcv": _as_tuple(getattr(self, "proj_out", None)),
            "K": int(getattr(self, "K", -1)),
            "sample_num": int(getattr(self, "sample_num", -1)),
        }
        print(f"[ARCH][PCVAdapter] {self._arch_sig}")

        # 可選：確定期望值後再打開斷言
        # assert self._arch_sig["c_pcv"] == 32, f"PCV proj_out changed: {self._arch_sig['c_pcv']}"

    def initialize(self, img_feat):
        n, c, h, w = img_feat.shape
        device = img_feat.device
        dtype = img_feat.dtype

        # 起始 mu（從 args 或類內屬性來）
        start = getattr(self, "init_mu", 0.0)
        # 轉換成 tensor，放同一個 device/dtype
        start = torch.as_tensor(start, device=device, dtype=dtype)
        # 若是多個 Gaussian，確保 shape = (K,)
        if start.ndim == 0:
            start = start.repeat(self.K)

        # 用同一個 device/dtype 建立座標
        coords0, coords1 = coords_grid(
            n, h, w, self.K, start_point=start, device=device, dtype=dtype)

        # 建立 sigma、w 也要在同一個 device/dtype
        sigma = torch.ones(n, self.K, h, w, device=device, dtype=dtype) * \
            (getattr(self, "init_sigma", 1.0) / (2 ** self.n_downsample))
        w = torch.full((n, self.K, h, w), 1.0 / float(self.K),
                       device=device, dtype=dtype)

        return coords0, coords1, sigma, w

    # 建議把 forward 介面改成：
    def forward(self, f4L, f4R, mf_08=None, motion_features_list=None, iters=3):
        if mf_08 is None:
            raise RuntimeError(
                "PCVAdapter.forward 需要 mf_08 (1/8 尺度的 motion/context 特徵)")
        B, _, H4, W4 = f4L.shape
        H8, W8 = H4 // 2, W4 // 2
        H16, W16 = H4 // 4, W4 // 4

        # 1/4 的 PPM 特徵降到 128ch 給 Corr
        fmap1 = self.f4_proj(f4L)    # [B,128,H/4,W/4]
        fmap2 = self.f4_proj(f4R)    # [B,128,H/4,W/4]

        # Corr
        corr_fn = CorrBlock1D(
            fmap1.float(), fmap2.float(),
            sample_num=self.args.sample_num,
            num_levels=self.args.corr_levels,
            downsample=self.args.n_downsample,
            compress_factor=self.compress_factor
        )

        # 初始化 (coords0/1, sigma, w)
        coords0, coords1, sigma, w = self.initialize(fmap1)

        # ---- 初始化三層 hidden states（依 hidden_dims）----
        net_list = [
            torch.tanh(fmap1.new_zeros(
                B, self.hidden_dims[0], H4,  W4)),  # 1/4
            torch.tanh(fmap1.new_zeros(
                B, self.hidden_dims[1], H8,  W8)),  # 1/8
            torch.tanh(fmap1.new_zeros(
                B, self.hidden_dims[2], H16, W16)),  # 1/16
        ]

        # ---- 準備三層 (z,q,r) 佔位（和 PCV 原本 API 對齊即可，全零即可）----
        def _zeros(ch, h, w): return fmap1.new_zeros(B, ch, h, w)
        inp_list = [
            [_zeros(self.hidden_dims[0], H4,  W4)] * 3,  # 1/4 : (z,q,r)
            [_zeros(self.hidden_dims[1], H8,  W8)] * 3,  # 1/8
            [_zeros(self.hidden_dims[2], H16, W16)] * 3,  # 1/16
        ]

        # 以傳給 updater 的 mf_08 為基準
        B, _, H8, W8 = mf_08.shape
        device = mf_08.device

        #  hidden 通道組態，跟 BasicMultiUpdateBlock 一致
        hdim04, hdim08, hdim16 = self.hidden_dims  # (128, 128, 128)

        # 1) 建立 net（注意都是 batch = B）
        net_list = [
            # 對齊 updater.gru04 的空間尺度
            torch.zeros(B, hdim04, H8 * 2, W8 * 2, device=device),  # 1/4
            torch.zeros(B, hdim08, H8,     W8,     device=device),  # 1/8
            torch.zeros(B, hdim16, H8 // 2, W8 // 2, device=device)  # 1/16
        ]

        # 2) 建立 inp（cz, cr, cq），形狀要與對應的 net[i] 一致
        def zero_like(shape): return torch.zeros(shape, device=device)
        inp = [
            (zero_like((B, hdim04, H8 * 2, W8 * 2)),
             zero_like((B, hdim04, H8 * 2, W8 * 2)),
             zero_like((B, hdim04, H8 * 2, W8 * 2))),  # for gru04

            (zero_like((B, hdim08, H8,     W8)),
             zero_like((B, hdim08, H8,     W8)),
             zero_like((B, hdim08, H8,     W8))),      # for gru08

            (zero_like((B, hdim16, H8 // 2, W8 // 2)),
             zero_like((B, hdim16, H8 // 2, W8 // 2)),
             zero_like((B, hdim16, H8 // 2, W8 // 2)))  # for gru16
        ]

        # 3)（optional）安全檢查：全部都要是同一個 batch
        assert net_list[1].shape[0] == B and net_list[2].shape[0] == B
        for triple in inp:
            for t in triple:
                assert t.shape[0] == B

        # 4) 呼叫 updater（保持原本的其他引數）

        coords1 = coords1.detach()
        corr = corr_fn(coords1, sigma, test_mode=False)
        mu = (coords0 - coords1).detach()
        # --- one-time shape check before calling updater ---
        if not hasattr(self, "_shape_checked"):
            try:
                print(
                    f"[SHAPE-CHECK][PCV] fmap1: {tuple(fmap1.shape)} fmap2: {tuple(fmap2.shape)}")
                print(f"[SHAPE-CHECK][PCV] mf_08: {tuple(mf_08.shape)}")
                print(
                    f"[SHAPE-CHECK][PCV] net_list[0]: {tuple(net_list[0].shape)} net_list[1]: {tuple(net_list[1].shape)} net_list[2]: {tuple(net_list[2].shape)}")
                # corr 是 CorrBlock1D(corr_fn) 返回的 callable；不印太深層，僅標記存在
                print(
                    f"[SHAPE-CHECK][PCV] corr_fn ready, K={self.K}, sample_num={self.sample_num}")
            except Exception as e:
                print("[SHAPE-CHECK][PCV] failed:", e)
            self._shape_checked = True

        # ====== 迭代更新 ======
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1, sigma, test_mode=False)
            # --- 將 corr 尺寸對齊 GRU-04 的 hidden（通常是 H/8×W/8）---
            if corr.shape[-2:] != net_list[0].shape[-2:]:
                corr = F.interpolate(
                    corr, size=net_list[0].shape[-2:], mode='bilinear', align_corners=True)
            mu = (coords0 - coords1).detach()

            net_list, up_mask, mu, sigma, w = self.updater(
                net_list, inp, corr, mu=mu, w=w, sigma=sigma,
                iter16=self.args.n_gru_layers >= 3,
                iter08=self.args.n_gru_layers >= 2,
                iter04=True,
                update=True,
                mf_08=mf_08   # <-- 加上這個
            )
            # --- 將 updater 回傳的 mu/sigma/w 空間尺寸，對齊 coords0 (通常是 1/4 尺度) ---
            if mu.ndim == 4 and mu.shape[-2:] != coords0.shape[-2:]:
                mu = F.interpolate(
                    mu, size=coords0.shape[-2:], mode='bilinear', align_corners=True)
            if sigma.ndim == 4 and sigma.shape[-2:] != coords0.shape[-2:]:
                sigma = F.interpolate(
                    sigma, size=coords0.shape[-2:], mode='bilinear', align_corners=True)
            if w.ndim == 4 and w.shape[-2:] != coords0.shape[-2:]:
                w = F.interpolate(
                    w, size=coords0.shape[-2:], mode='bilinear', align_corners=True)
            coords1 = coords0 - mu

        feat = torch.cat([w, mu, sigma], dim=1)   # [B,3K,H/4,W/4]
        return self.proj(feat)                    # -> [B,proj_out,H/4,W/4]
