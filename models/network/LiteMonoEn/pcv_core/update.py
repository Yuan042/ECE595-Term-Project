import torch
import torch.nn as nn
import torch.nn.functional as F


# --- add begin: light ResidualBlock used by conv2 ---


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm='instance', stride=1):
        super().__init__()
        # 投影捷徑：通道或stride不一致時用1x1 conv對齊
        self.proj = nn.Identity() if (in_ch == out_ch and stride == 1) else nn.Conv2d(
            in_ch, out_ch, 1, stride=stride, bias=False)
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

        if norm == 'instance':
            Norm = nn.InstanceNorm2d
        elif norm == 'batch':
            Norm = nn.BatchNorm2d
        else:
            Norm = None

        self.norm1 = Norm(out_ch) if Norm is not None else nn.Identity()
        self.norm2 = Norm(out_ch) if Norm is not None else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.proj(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.act(x + shortcut)
        return x


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)  # correlation+flow
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + cq)

        h = (1 - z) * h + z * q
        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args

        cor_planes = args.sample_num * args.corr_levels  # 27
        self.convc1 = nn.Conv2d(cor_planes, 64, 3, padding=1)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convc3 = nn.Conv2d(64, 48, 3, padding=1)
        self._param_proj = nn.ModuleDict()
        self.param_out_ch = 64
        self.convf2 = nn.Conv2d(
            self.param_out_ch, self.param_out_ch - 3 * args.gauss_num, 3, padding=1)

    def forward(self, disp, corr, w, sigma):
        N, C, H, W = corr.shape
        corr = corr.reshape(N, self.args.corr_levels, self.args.gauss_num, self.args.sample_num, H, W).permute(0, 2, 1,
                                                                                                               3, 4, 5)
        corr = corr.reshape(-1, self.args.corr_levels *
                            self.args.sample_num, H, W)
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        cor = F.relu(self.convc3(cor))
        cor = cor.reshape(N, -1, H, W)
        param = torch.cat((disp, w.detach(), sigma.detach()), dim=1)
        key = f"param_proj_{param.shape[1]}"
        if key not in self._param_proj:
            self._param_proj[key] = nn.Conv2d(
                param.shape[1], self.param_out_ch, 7, padding=3).to(param.device)
        param_f = F.relu(self._param_proj[key](param))
        param_f = F.relu(self.convf2(param_f))
        return torch.cat([cor, param_f, param], dim=1)


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)


def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class ParametersUpdater(nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super(ParametersUpdater, self).__init__()
        self.args = args
        self.head = FlowHead(input_dim, hidden_dim, args.gauss_num)
        self.register_buffer('gamma1', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('gamma2', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('gamma3', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('sigma0', torch.tensor(0.5, dtype=torch.float32))
        self.register_buffer('eps',    torch.tensor(1e-6, dtype=torch.float32))
        self.w_clip = 3.0  # 用 float

    def forward(self, hidden_state, mu, sigma, w):

        H, W = hidden_state.shape[-2:]
        if sigma.ndim >= 4 and sigma.shape[-2:] != (H, W):
            sigma = F.interpolate(sigma, size=(
                H, W), mode='bilinear', align_corners=True)
        if w.ndim >= 4 and w.shape[-2:] != (H, W):
            w = F.interpolate(w, size=(H, W), mode='bilinear',
                              align_corners=True)
        if mu.ndim >= 4 and mu.shape[-2:] != (H, W):
            mu = F.interpolate(
                mu, size=(H, W), mode='bilinear', align_corners=True)

        delta = self.head(hidden_state)
        _, M, _, _ = delta.shape

        def _resize_like(x, ref):
            # 保留 dtype/device 並對齊空間解析度
            if x.ndim >= 4 and x.shape[-2:] != ref.shape[-2:]:
                return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=True)
            return x

        # 將 sigma0 轉成張量並擴展/插值到與 sigma 一樣的形狀
        sigma0 = self.sigma0.to(sigma.device, dtype=sigma.dtype).view(
            1, 1, 1, 1).expand(sigma.size(0), 1, *sigma.shape[-2:])
        if not torch.is_tensor(sigma0):
            sigma0 = torch.tensor(
                float(sigma0), device=sigma.device, dtype=sigma.dtype)
        if sigma0.ndim == 0:
            sigma0 = sigma0.view(1, 1, 1, 1)
        # 先擴展到通道與 batch 可廣播，再做空間尺寸對齊
        sigma0 = sigma0.expand(sigma.size(0), 1, *sigma.shape[-2:])
        sigma0 = _resize_like(sigma0, sigma)

        # 將 M、w 也對齊到 sigma 的空間尺寸

        w = _resize_like(w, sigma)
        M_t = torch.tensor(float(self.args.gauss_num),
                           device=sigma.device, dtype=sigma.dtype)

        # feed forward gradients
        delta_sigma = 0.5 * (((1 - M_t * w) * sigma ** 2 - sigma0 ** 2 -
                             delta ** 2) / (M_t * sigma ** 3) + w * sigma / (self.sigma0 ** 2))
        delta_mu = -0.5 * delta * \
            (1 / (M * sigma ** 2) + w / (self.sigma0 ** 2))
        beta = 0.5 * (-1 / (M * w + self.eps) + torch.log(self.sigma0 * M * w /
                      sigma + self.eps) + (sigma ** 2 + delta ** 2) / (2 * self.sigma0 ** 2) + 0.5)
        delta_w = beta - torch.sum(beta, dim=1, keepdim=True) / M

        # clip the gradients
        delta_sigma = torch.clamp(
            delta_sigma * self.gamma1, min=-3.0,   max=3.0)
        delta_mu = torch.clamp(delta_mu * self.gamma2, min=-128.0, max=128.0)
        delta_w = torch.clamp(delta_w * self.gamma3,
                              min=-(1.0/(M_t*4.0)), max=(1.0/(M_t*4.0)))

        # update & clip the parameters
        sigma = torch.clamp(sigma - delta_sigma, min=0.1, max=16.0)
        mu = mu - delta_mu
        w = torch.clamp(w - delta_w,         min=0.0, max=1.0)
        # normalize
        w = w / torch.sum(w, dim=1, keepdim=True)
        return mu, w, sigma


class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=(128, 128, 128),
                 encoder_output_dim=384, encoder_output_dim_04=None,
                 encoder_output_dim_08=None, encoder_output_dim_16=None):
        super().__init__()
        self.args = args
        self.hidden_dims = hidden_dims

        # 這個模塊的“標準化介面通道”：一律用 128 餵 GRU
        self.encoder_output_dim = 128

        # 將 1/8 的 motion/context 先壓到 256，再投到 128 當 GRU 的 ctx
        self.conv2 = nn.Sequential(
            ResidualBlock(self.encoder_output_dim, 128, 'instance', stride=1),
            nn.Conv2d(128, 256, 3, padding=1),               # -> 256
            nn.ReLU(inplace=True)
        )
        self.conv2_out = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False), nn.ReLU(
                inplace=True)  # 256 -> 128
        )

        # 1/16 分支：從 1/8 之 128 做一次 stride=2 下采樣再還原成 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(
                inplace=True)  # -> H/16, 256
        )
        self.conv3_out = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False), nn.ReLU(
                inplace=True)  # 256 -> 128
        )

        # 三層 GRU（04/08/16），每層 input_dim = 128(+ skip 128) 的總和已在 forward 控制
        self.gru04 = ConvGRU(
            # 128 in
            hidden_dims[0], input_dim=self.encoder_output_dim)
        self.gru08 = ConvGRU(
            # 128 + 128
            hidden_dims[1],
            input_dim=self.encoder_output_dim + hidden_dims[0] + hidden_dims[2])
        self.gru16 = ConvGRU(
            # 128 + 128
            hidden_dims[2], input_dim=self.encoder_output_dim + hidden_dims[1])

        # 上采樣 mask 用 1/4 層 hidden
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[0], 256, 3,
                      padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1)
        )

        # 參數更新頭
        self.ParametersUpdater = ParametersUpdater(
            self.args, input_dim=128, hidden_dim=256)

        # 小工具：動態 1x1 投影，確保輸入通道對齊
        self._proj_cache = nn.ModuleDict()
        # 預先註冊（對齊 ckpt：mf_08 是 256ch → 128）
        self._proj_cache["mf_to128"] = nn.Conv2d(
            256, 128, kernel_size=1, bias=False)
        # self._proj_cache["mf_to128_04_like"] = nn.Conv2d(
        #     256, 128, kernel_size=1, bias=False)

    def _project_to(self, x, target_ch: int, key: str):
        in_ch = x.shape[1]
        if in_ch == target_ch:
            return x
        if (key not in self._proj_cache
            or self._proj_cache[key].in_channels != in_ch
                or self._proj_cache[key].out_channels != target_ch):
            self._proj_cache[key] = nn.Conv2d(
                in_ch, target_ch, kernel_size=1, bias=False).to(x.device)
        return self._proj_cache[key](x)

    def forward(self, net, inp, corr=None, mu=None, w=None, sigma=None,
                iter04=True, iter08=True, iter16=True, update=True,
                test_mode=False, motion_features_list=None, mf_08=None):

        # 1) 必須有 1/8 的 motion/context，先把它投到 128
        if mf_08 is None:
            raise RuntimeError(
                "PCV update: mf_08 is None. 請在 PCVAdapter 呼叫時以 mf_08=... 顯式傳 1/8 motion features。")

        mf = self._project_to(mf_08, self.encoder_output_dim,
                              key="mf_to128")  # -> [B,128,H/8,W/8]

        # 2) 依序做 1/8 與 1/16 的 context
        feat08 = self.conv2(mf)          # [B,256,H/8,W/8]
        ctx08 = self.conv2_out(feat08)  # [B,128,H/8,W/8]

        feat16 = self.conv3(mf)          # [B,256,H/16,W/16]
        ctx16 = self.conv3_out(feat16)  # [B,128,H/16,W/16]

        # 3) 先更新 1/16
        if iter16:
            net[2] = self.gru16(
                net[2], *(inp[2]),
                ctx16,                   # 128
                pool2x(net[1])           # 128
            )

        # 4) 再更新 1/8
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), ctx08,
                                    pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), ctx08, pool2x(net[0]))

        # 5) 最後更新 1/4
        if iter04:
            net[0] = self.gru04(
                net[0], *(inp[0]),
                interp(self._project_to(mf, 128, "mf_to128_04_like"), net[0])
            )

        if not update:
            # 回傳 minimal 的 motion features
            return net, [ctx08, ctx16]

        # 6) 參數更新
        mu, w, sigma = self.ParametersUpdater(net[0], mu, sigma, w)
        up_mask = self.mask(net[0])

        return net, up_mask, mu, sigma, w
