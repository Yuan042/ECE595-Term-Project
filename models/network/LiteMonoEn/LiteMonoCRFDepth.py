# Written by Ukcheol Shin (shinwc159@gmail.com)
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.network.LiteMonoEn.submodule import build_l1c_volume
import random
import math
import os

from models.network.LiteMonoEn.third_party.litemono.networks.depth_encoder import LiteMono as DepthNetEncoder
from .LiteMonoEn import LiteMonoEnBackbone
from .pcv_module import PCVAdapter
from .uper_crf_head import PSP
from .newcrf_layers import NewCRF


class MonoStereoCRFDepth(nn.Module):
    def __init__(self,
                 maxdisp: int,
                 use_concat_volume: bool = False,
                 use_3d_decoder: bool = False,
                 encoder_model: str = "tiny",
                 pre_trained: bool = False,
                 ckpt_path: str | None = None,
                 disp_head: str = "classify",
                 # ↓↓↓ 這些是 YAML 可覆寫的關鍵參數，直接由外部傳進來
                 in_chans: int = 3,
                 ppm_channels: int = 512,
                 pcv_out: int = 32,
                 **kwargs):
        super().__init__()
        print("[PIPELINE] EnLiteMono+PSP+PCV(1/4)+NeWCRF is ACTIVE")

        # === 固定存成成員 ===
        self.maxdisp = int(maxdisp)
        self.use_concat_volume = bool(use_concat_volume)
        self.use_3d_decoder = bool(use_3d_decoder)
        self.encoder_model = str(encoder_model)
        self.pre_trained = bool(pre_trained)
        self.ckpt_path = ckpt_path
        self.disp_head = str(disp_head)

        # === 取代原本 opt.model.* 的四個屬性 ===
        self.in_chans = int(in_chans)
        self.c_ppm = int(ppm_channels)
        self.c_pcv = int(pcv_out)
        # --- NEW: 決定 disparity head 模式（供後續 CRF / head 使用） ---
        _disp = (self.disp_head or "classify").lower()
        if _disp in ("classify", "classification", "cls"):
            self.disp_head_mode = "classify"
        elif _disp in ("regress", "regression", "reg"):
            self.disp_head_mode = "regression"
        else:
            # 不認得就回退到分類（和原 MSCRF 設定一致）
            self.disp_head_mode = "classify"

        norm_cfg = dict(type='BN', requires_grad=True)

        # 例如 [C4, C8, C16, C16]
        # 1) 建立 LiteMono 編碼器封裝
        self.feat_encoder = LiteMonoEnBackbone(
            in_chans=self.in_chans, model=self.encoder_model
        )
        enc_dims = getattr(self.feat_encoder.encoder, "dims", None)
        if enc_dims is None or len(enc_dims) < 3:
            raise RuntimeError(f"LiteMono encoder.dims 無效：{enc_dims}")

        c4, c8, c16 = enc_dims[:3]
        c32 = c16  # 補第 4 層：用 f16 再下採樣的通道數，設定為與 C16 一致（穩定做法）
        # 統一存成成員，後面不要再用未定義的 in_channels
        self.in_chs4 = [c4, c8, c16, c32]

        # PSP / UPer Head 需要四個尺度的 in_channels（list/tuple）
        decoder_cfg = dict(
            in_channels=self.in_chs4,  # <— 關鍵：要是 list/tuple，且長度=4
            in_index=[0, 1, 2, 3],           # 對應上面四個尺度的索引
            # input_transform="multiple_select",
            channels=self.c_ppm,             # 原來設定的中間通道（例如 512）
            num_classes=self.c_pcv,          # 這裡通常是 head 的輸出通道：要餵 PCV/NeWCRF 的維度
            # enable_crf=False                 # 如果用的是純 PSP 對齊 head（非 CRF），就 False
        )
        self.decoder = PSP(**decoder_cfg)

        self.c_pcv = 32
        ctx04 = int(self.c_ppm + self.c_pcv)   # 例如 512 + 32 = 544
        # 1/8 的 motion(256) + 下層 hidden(128)
        ctx08 = 256 + 128
        ctx16 = 256 + 128
        self.pcv = PCVAdapter(K=3, proj_out=self.c_pcv, hidden_dims=(128, 128, 128),
                              ctx04_channels=ctx04,
                              ctx08_channels=ctx08,
                              ctx16_channels=ctx16)  # 只在 1/4 尺度用
        # 取得 encoder 在 1/8 尺度的通道數（依你的 LiteMono tiny/large 等）
        c8 = 128  # 若 LiteMono 1/8 輸出是 64，這裡就改 64；之後會被 1x1 conv 投到 256

        # 將 1/8 尺度的特徵壓到 PCV 期望的 256ch
        try:
            # DepthNetEncoder 定義的 [C4, C8, C16]
            c8_in = int(self.feat_encoder.encoder.num_ch_enc[1])
        except Exception:
            c8_in = 64  # 保底（tiny）
        self.mf08_proj = nn.Conv2d(c8_in, 256, kernel_size=1, bias=False)

        self._pipeline_tag = "EnLiteMono+PSP+PCV(1/4)+NeWCRF"
        print(f"[PIPELINE] {self._pipeline_tag} | encoder={self.encoder_model} | psp_channels={getattr(self, 'psp_channels', 512)} | pcv_out={self.c_pcv}")

        self.costvolume_builter = self._forbid_costvolume

        v_dim = decoder_cfg['num_classes']*4
        self.win_sizes = 7
        self.crf_dims = [self.c_pcv, self.c_pcv, self.c_pcv, self.c_pcv]
        # ← 變成 [C4, C8, C16, C32]，tiny 例子是 [32,64,128,128]
        self.v_dims = list(self.in_chs4)
        # === Debug: log & assert auto-derived dims (LiteMonoCRFDepth) ===

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
            "v_dims": _as_tuple(getattr(self, "v_dims", None)),
            "c_pcv": _as_tuple(getattr(self, "c_pcv", None)),
        }
        print(f"[ARCH][LiteMonoCRFDepth] {self._arch_sig}")
        self.crf3 = NewCRF(
            input_dim=self.in_chs4[3], embed_dim=self.crf_dims[3], window_size=self.win_sizes, v_dim=self.v_dims[3], num_heads=32)
        self.crf2 = NewCRF(
            input_dim=self.in_chs4[2]+12, embed_dim=self.crf_dims[2], window_size=self.win_sizes, v_dim=self.v_dims[2], num_heads=16)
        self.crf1 = NewCRF(
            input_dim=self.in_chs4[1]+24, embed_dim=self.crf_dims[1], window_size=self.win_sizes, v_dim=self.v_dims[1], num_heads=8)
        self.crf0 = NewCRF(
            input_dim=self.in_chs4[0]+48 + self.c_pcv, embed_dim=self.crf_dims[0], window_size=self.win_sizes, v_dim=self.v_dims[0], num_heads=4)

        self.disp_head1 = DispHead(
            input_dim=self.crf_dims[0], mode=self.disp_head_mode, max_disp=self.maxdisp)
        self.disp_head2 = DispHead(
            input_dim=self.crf_dims[1], mode=self.disp_head_mode, max_disp=self.maxdisp//2)
        self.disp_head3 = DispHead(
            input_dim=self.crf_dims[2], mode=self.disp_head_mode, max_disp=self.maxdisp//4)
        self.disp_head4 = DispHead(
            input_dim=self.crf_dims[3], mode=self.disp_head_mode, max_disp=self.maxdisp//4)
        if self.disp_head_mode == 'regression':
            # the output is a value between range 0-1, after upsample original resolution, then multiply 192
            self.scale_factor = [4, 8, 16, 32]
        elif self.disp_head_mode == 'classify':
            # the output channel is 48, 48*4 ==> 192 disp channel
            self.scale_factor = [4, 2, 1, 1]

        if pre_trained == False:
            self.init_weights(pretrained=None)
        else:
            self.init_weights(pretrained=ckpt_path)

        self.c32 = self.in_chs4[3]           # 128
        self.v_proj32 = nn.Conv2d(
            self.c_ppm + self.c_pcv, self.v_dims[3], kernel_size=1, bias=False)
        # ---- add: predefine all projection layers (match ckpt keys) ----
        # multi-scale back-projection (stereo path)
        # multi-scale back-projection (stereo path)  ← 改：in_channels = 8
        self.v_proj16_from_e3 = nn.Conv2d(
            8, self.v_dims[2], kernel_size=1, bias=False)  # 8 → 128
        self.v_proj08_from_e2 = nn.Conv2d(
            8, self.v_dims[1], kernel_size=1, bias=False)  # 8 → 64
        self.v_proj04_from_e1 = nn.Conv2d(
            8, self.v_dims[0], kernel_size=1, bias=False)  # 8 → 32

        # mono path (names must match checkpoint)   ← 改：in_channels = 8
        self.v_proj32_m = nn.Conv2d(
            # 544 → 128（不變）
            self.c_ppm + self.c_pcv, self.v_dims[3], kernel_size=1, bias=False)
        self.v_proj16_m_from_e3 = nn.Conv2d(
            8, self.v_dims[2], kernel_size=1, bias=False)  # 8 → 128
        self.v_proj08_m_from_e2 = nn.Conv2d(
            8, self.v_dims[1], kernel_size=1, bias=False)  # 8 → 64
        self.v_proj04_m_from_e1 = nn.Conv2d(
            8, self.v_dims[0], kernel_size=1, bias=False)  # 8 → 32

        # ---- end add ----

    def _project_to(self, x: torch.Tensor, out_ch: int, name: str) -> torch.Tensor:
        """
        將張量 x 以 1x1 conv 投影到 out_ch 通道。
        若快取的層 in/out 通道不匹配，就自動重建（並覆蓋到 self.<name>）。
        """
        layer = getattr(self, name, None)
        need_rebuild = (
            layer is None
            or not isinstance(layer, nn.Conv2d)
            or layer.weight.shape[1] != x.shape[1]
            or layer.weight.shape[0] != out_ch
        )
        if need_rebuild:
            layer = nn.Conv2d(x.shape[1], out_ch,
                              kernel_size=1, bias=False).to(x.device)
            setattr(self, name, layer)
        return layer(x)

    def _posenc(self, ref: torch.Tensor, k: int) -> torch.Tensor:
        """
        產生 k 通道的 2D 位置編碼，與 ref 有相同的 (B,H,W)。
        做法：用 [-1,1] 的 x,y 網格搭配多頻 sin/cos 展開，湊到 k 通道。
        """
        B, _, H, W = ref.shape
        device, dtype = ref.device, ref.dtype

        ys = torch.linspace(-1, 1, H, device=device,
                            dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        xs = torch.linspace(-1, 1, W, device=device,
                            dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)

        enc = [xs, ys]  # 先放 2 個通道
        if k <= 2:
            return torch.cat(enc[:k], dim=1)

        # 用多頻率的 sin/cos 擴張到 k 通道
        freqs = (1.0, 2.0, 4.0, 8.0, 16.0)
        for f in freqs:
            if len(enc) >= k:
                break
            enc.append(torch.sin(f * math.pi * xs))
            if len(enc) >= k:
                break
            enc.append(torch.cos(f * math.pi * xs))
            if len(enc) >= k:
                break
            enc.append(torch.sin(f * math.pi * ys))
            if len(enc) >= k:
                break
            enc.append(torch.cos(f * math.pi * ys))
        # 若還不夠就重複切頻
        while len(enc) < k:
            enc.append(xs)
            if len(enc) < k:
                enc.append(ys)

        return torch.cat(enc[:k], dim=1)

    def _adapt_in(self, x: torch.Tensor) -> torch.Tensor:
        # 將輸入通道數對齊到 self.in_chans
        c_in = x.size(1)
        if c_in == self.in_chans:
            return x
        if c_in == 3 and self.in_chans == 1:
            return x.mean(dim=1, keepdim=True)          # 3→1
        if c_in == 1 and self.in_chans == 3:
            return x.repeat(1, 3, 1, 1)                  # 1→3
        # 其他情形：保險 1x1 conv 對齊
        if not hasattr(self, "_fallback_proj"):
            self._fallback_proj = torch.nn.Conv2d(
                c_in, self.in_chans, kernel_size=1, bias=False)
        return self._fallback_proj(x)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        # LiteMonoEnBackbone 目前沒有 init_weights
        if hasattr(self.feat_encoder, "init_weights"):
            self.feat_encoder.init_weights(pretrained=pretrained)
        # PSP decoder 不一定有 init_weights
        if hasattr(self.decoder, "init_weights"):
            self.decoder.init_weights()

    def _forbid_costvolume(self, *args, **kwargs):
        raise RuntimeError(
            "Old L1C/GWC cost-volume Shouldn't be called, current version: PCV@1/4。")

    def build_2d_volume(self, feat_l, feat_r, scale=4):
        # Bx D/4 x H/4 x W/4
        # candidate information
        raise RuntimeError("build_2d_volume 已停用；目前僅使用 PCV@1/4")
        if self.use_swin_backbone:
            l1c_volume = build_l1c_volume(
                feat_l, feat_r, self.maxdisp // scale)
        else:
            l1c_volume = build_l1c_volume(
                feat_l["gwc_feature"], feat_r["gwc_feature"], self.maxdisp // 4)

        # Bx 1 x H/4 x W/4
        # initial candidate
        if self.use_swin_backbone:
            init_disp = l1c_volume.argmin(dim=1, keepdim=True)
            # 256+48+1
            # volume = torch.cat((feat_l, init_disp, l1c_volume), 1)
        else:
            init_disp = l1c_volume.argmin(dim=1, keepdim=True)
            # 320+48+1
            # volume = torch.cat((feat_l["gwc_feature"], init_disp, l1c_volume), 1)

        return l1c_volume

    def forward(self, left, right):
        left = self._adapt_in(left)
        right = self._adapt_in(right) if right is not None else None

        # 1) 取四層特徵
        feats_left = self.feat_encoder(left)   # [f4,f8,f16,f32]
        mf_08 = self.mf08_proj(feats_left[1])  # (B, 256, H/8, W/8)
        feats_right = self.feat_encoder(right)  # [f4,f8,f16,f32]

        # 2) PSP：回一個 1/4 聚合特徵
        agg_ppm_left = self.decoder(feats_left)
        agg_ppm_right = self.decoder(feats_right)
        L, R = agg_ppm_left, agg_ppm_right
        assert L.dim() == 4 and R.dim() == 4, "PCV 需要 4D 張量 (B,C,H,W)"
        _, _, H4, W4 = L.shape
        if W4 < 2:
            # 若太小，先放大（只影響進 PCV 的分支，不動其他路徑）
            L = F.interpolate(L, scale_factor=2,
                              mode="bilinear", align_corners=False)
            R = F.interpolate(R, scale_factor=2,
                              mode="bilinear", align_corners=False)

        # [B, C_pcv, H/4(或更大), W/4(或更大)]
        pcv_feat = self.pcv(L, R, mf_08)
        # 供 level-3 用的 "v"（原本把1/4餵到最上層CRF）

        agg_1_4 = torch.cat([agg_ppm_left, pcv_feat], dim=1)

        # --- one-time shape check before projecting v3 / feeding CRF ---
        if not hasattr(self, "_shape_checked"):
            try:
                to_check = [
                    ("agg_ppm_left", agg_ppm_left),
                    ("pcv_feat", pcv_feat),
                ]
                for name, t in to_check:
                    if isinstance(t, torch.Tensor):
                        print(
                            f"[SHAPE-CHECK][LiteMonoCRFDepth] {name}: {tuple(t.shape)}")
                if isinstance(pcv_feat, torch.Tensor) and hasattr(self, "c_pcv"):
                    assert pcv_feat.shape[1] == int(self.c_pcv), \
                        f"pcv_feat C={pcv_feat.shape[1]} vs c_pcv={self.c_pcv}"
            except Exception as e:
                print("[SHAPE-CHECK][LiteMonoCRFDepth] failed:", e)
            self._shape_checked = True

        # 64 -> 128
        v3 = self._project_to(agg_1_4, self.v_dims[3], "v_proj32")
        # ---- Level-3（1/32）----
        e3 = self.crf3(feats_left[3], v3)

        if os.environ.get("DBG_ONCE") == "1":
            print("[DBG] f32:", feats_left[3].shape, "v3:", v3.shape)
            raise SystemExit(0)  # 只跑到這裡就退出，不進入訓練

        if self.training:
            d3 = self.disp_head4(e3, self.scale_factor[3])

        # ---- Level-2（1/16）：不再建 volume，直接吃 feats + 來自上一層的 v ----
        e3 = nn.PixelShuffle(2)(e3)
        feat = feats_left[2]
        feat = torch.cat([feat, self._posenc(feat, 12)], dim=1)   # +12 通道位置編碼
        # 8ch -> v_dim(通常128)
        v2 = self._project_to(e3, self.v_dims[2], "v_proj16_from_e3")
        # (x=feat, v=v2)
        e2 = self.crf2(feat, v2)

        if self.training:
            d2 = self.disp_head3(e2, self.scale_factor[2])

        # ---- Level-1（1/8）----
        e2 = nn.PixelShuffle(2)(e2)
        feat = feats_left[1]
        feat = torch.cat([feat, self._posenc(feat, 24)], dim=1)   # +24 通道位置編碼
        # 8ch -> v_dim(通常128或64)
        v1 = self._project_to(e2, self.v_dims[1], "v_proj08_from_e2")
        e1 = self.crf1(feat, v1)

        if self.training:
            d1 = self.disp_head2(e1, self.scale_factor[1])

        # ---- Level-0（1/4）：把 pcv_feat 串回 inputs，注意 crf0.input_dim 已 + C_pcv ----
        e1 = nn.PixelShuffle(2)(e1)
        if pcv_feat.shape[-2:] != feats_left[0].shape[-2:]:
            pcv_feat = F.interpolate(pcv_feat, size=feats_left[0].shape[-2:],
                                     mode="bilinear", align_corners=True)
        feat_1_4_base = torch.cat([feats_left[0], pcv_feat], dim=1)
        feat_1_4 = torch.cat([feat_1_4_base, self._posenc(
            feats_left[0], 48)], dim=1)  # +48 通道位置編碼
        # 8ch -> v_dim(通常32)
        v0 = self._project_to(e1, self.v_dims[0], "v_proj04_from_e1")
        e0 = self.crf0(feat_1_4, v0)

        d0 = self.disp_head1(e0, self.scale_factor[0])

        disp_stereo = [d3, d2, d1, d0] if self.training else d0

        # 4) Mono 分支：把「右」改成「左自身」，同樣只在 1/4 做 PCV
        pcv_feat_m = self.pcv(agg_ppm_left, agg_ppm_left, mf_08)
        agg_1_4_m = torch.cat([agg_ppm_left, pcv_feat_m], dim=1)

        v3m = self._project_to(agg_1_4_m, self.v_dims[3], "v_proj32_m")
        e3 = self.crf3(feats_left[3], v3m)
        if self.training:
            d3 = self.disp_head4(e3, self.scale_factor[3])

        e3 = nn.PixelShuffle(2)(e3)
        v2m = self._project_to(e3, self.v_dims[2], "v_proj16_m_from_e3")
        x2m = torch.cat([feats_left[2], self._posenc(
            feats_left[2], 12)], dim=1)  # 128+12=140
        e2 = self.crf2(x2m, v2m)  # (x, v)
        if self.training:
            d2 = self.disp_head3(e2, self.scale_factor[2])

        e2 = nn.PixelShuffle(2)(e2)
        v1m = self._project_to(e2, self.v_dims[1], "v_proj08_m_from_e2")
        x1m = torch.cat([feats_left[1], self._posenc(
            feats_left[1], 24)], dim=1)  # 64/128 + 24
        e1 = self.crf1(x1m, v1m)  # (x, v)
        if self.training:
            d1 = self.disp_head2(e1, self.scale_factor[1])

        if pcv_feat_m.shape[-2:] != feats_left[0].shape[-2:]:
            pcv_feat_m = F.interpolate(
                pcv_feat_m, size=feats_left[0].shape[-2:], mode="bilinear", align_corners=True)
        feat_1_4_m = torch.cat([feats_left[0], pcv_feat_m], dim=1)
        feat_1_4_m = torch.cat(
            [feat_1_4_m, self._posenc(feats_left[0], 48)], dim=1)  # +48
        e1 = nn.PixelShuffle(2)(e1)
        v0m = self._project_to(e1, self.v_dims[0], "v_proj04_m_from_e1")
        e0 = self.crf0(feat_1_4_m, v0m)  # (x, v)
        d0 = self.disp_head1(e0, self.scale_factor[0])
        disp_mono_l = [d3, d2, d1, d0] if self.training else d0

        return disp_mono_l, disp_stereo

    def forward_mono(self, left):
        feats_left = self.feat_encoder(left)        # [f4,f8,f16,f32]
        agg_ppm_left = self.decoder(feats_left)     # 1/4 聚合語義
        pcv_feat_m = self.pcv(
            agg_ppm_left, agg_ppm_left) if self.pcv is not None else 0
        agg_1_4_m = torch.cat([agg_ppm_left, pcv_feat_m], dim=1) if isinstance(
            pcv_feat_m, torch.Tensor) else agg_ppm_left

        e3 = self.crf3(feats_left[3], agg_1_4_m)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(feats_left[2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(feats_left[1], e2)
        e1 = nn.PixelShuffle(2)(e1)
        feat_1_4_m = torch.cat([feats_left[0], pcv_feat_m], dim=1) if isinstance(
            pcv_feat_m, torch.Tensor) else feats_left[0]
        e0 = self.crf0(feat_1_4_m, e1)
        d0 = self.disp_head1(e0, self.scale_factor[0])
        return d0


class DispHead(nn.Module):
    def __init__(self, input_dim=128, mode='regression', max_disp=192):
        super(DispHead, self).__init__()
        self.mode = mode

        if self.mode == 'regression':
            self.head = self.regressor(in_channel=input_dim, out_channel=1)
            self.forward = self.forward_regression
        elif self.mode == 'classify':
            self.head = self.classifer(in_channel=input_dim, out_channel=48)
            self.forward = self.forward_classification
        else:
            print('Unsupported type!')

        self.max_disp = max_disp

    def convbn(self, in_channels, out_channels, kernel_size, stride, pad, dilation):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                       padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                             nn.BatchNorm2d(out_channels))

    def classifer(self, in_channel, out_channel):
        return nn.Sequential(self.convbn(in_channel, in_channel, 3, 1, 1, 1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False))

    def regressor(self, in_channel, out_channel=1):
        return nn.Sequential(self.convbn(in_channel, in_channel, 3, 1, 1, 1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                       padding=1, stride=1, bias=False),
                             nn.Sigmoid())  # 0-1 range

    def upsample(self, x, scale_factor=2, mode="bilinear", align_corners=False):
        """Upsample input tensor by a factor of 2
        """
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    def disparity_regression(self, x, maxdisp):
        assert len(x.shape) == 4
        disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
        disp_values = disp_values.view(1, maxdisp, 1, 1)
        return torch.sum(x * disp_values, 1, keepdim=False)

    def forward_regression(self, x, scale):
        # x = self.relu(self.norm1(x))
        disp = self.head(x)
        if scale > 1:
            disp = self.upsample(disp, scale_factor=scale)
        return disp.squeeze(1) * self.max_disp

    def forward_classification(self, x, scale):
        # x = self.relu(self.norm1(x))
        x = self.head(x)
        if scale > 1:
            x = self.upsample(x.unsqueeze(
                1), scale_factor=scale, mode='trilinear')
            x = torch.squeeze(x, 1)
        pred = F.softmax(x, dim=1)
        disp = self.disparity_regression(pred, self.max_disp)
        return disp
