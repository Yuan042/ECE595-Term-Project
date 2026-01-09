import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd //
                         2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    if H > 1:
        ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


# def coords_grid(batch, ht, wd):
#     coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
#     coords = torch.stack(coords[::-1], dim=0).float()
#     return coords[None].repeat(batch, 1, 1, 1)  # [N,2,H,W]

def coords_grid(n, h, w, gauss_num, start_point=None, device=None, dtype=None):
    import torch

    # 決定 device/dtype
    if isinstance(start_point, torch.Tensor):
        device = start_point.device if device is None else device
        dtype = start_point.dtype if dtype is None else dtype
    else:
        device = torch.device("cpu") if device is None else device
        dtype = torch.float32 if dtype is None else dtype

    # 建 1D 座標（在目標 device/dtype）
    x_coords = torch.arange(w, device=device, dtype=dtype).view(
        1, 1, 1, w)  # (1,1,1,W)
    y_coords = torch.arange(h, device=device, dtype=dtype).view(
        1, 1, h, 1)  # (1,1,H,1)

    # 目標 shape: (N, gauss_num, H, W)
    x_coords = x_coords.expand(n, gauss_num, h, w)
    y_coords = y_coords.expand(n, gauss_num, h, w)

    # 處理 start_point（必須同 device/dtype）
    if start_point is None:
        sp = torch.zeros(gauss_num, device=device, dtype=dtype)
    else:
        sp = torch.as_tensor(start_point, device=device,
                             dtype=dtype).view(gauss_num)

    # disparity 初始化：x - mu（每個 Gaussian component 一個 mu）
    x_coords = x_coords - sp.view(1, gauss_num, 1, 1)

    return x_coords, x_coords  # coords0, coords1


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def gauss_blur(input, N=5, std=1):
    B, D, H, W = input.shape
    x = torch.arange(N, device=device, dtype=torch.float32) - N//2
    y = torch.arange(N, device=device, dtype=torch.float32) - N//2
    x, y = torch.meshgrid(x, y, indexing='ij')
    unnormalized_gaussian = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * std ** 2))
    weights = unnormalized_gaussian / unnormalized_gaussian.sum().clamp(min=1e-4)
    weights = weights.view(1, 1, N, N).to(input)
    output = F.conv2d(input.reshape(B * D, 1, H, W), weights, padding=N // 2)
    return output.view(B, D, H, W)
