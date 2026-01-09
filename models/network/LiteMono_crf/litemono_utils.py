import torch
import torch.nn as nn
import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='bilinear',
           align_corners=False,
           **kwargs):
    """
    簡單包一層 F.interpolate，給 LiteMono_crf 用的 resize 工具。
    """
    return F.interpolate(input,
                         size=size,
                         scale_factor=scale_factor,
                         mode=mode,
                         align_corners=align_corners)


def normal_init(module, mean=0.0, std=1.0, bias=0.0):
    """
    權重初始化工具。就算不呼叫也沒關係，呼叫到就 normal_ 。
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean=mean, std=std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def load_checkpoint(model,
                    filename=None,
                    map_location="cpu",
                    strict=False,
                    logger=None,
                    **kwargs):
    """
    給 FLOP / timing 用的「安全版」load_checkpoint：

    - 如果 filename 是 None 或空字串：直接略過。
    - 如果檔案不存在：印一行提示，然後略過，不丟錯。
    - 如果 mmcv 存在且檔案真的在，就正常載入。
    """
    if not filename:
        print("[LiteMono_crf] load_checkpoint: 沒有指定 checkpoint，略過載入。")
        return None

    try:
        from mmcv.runner import load_checkpoint as mmcv_load_checkpoint
    except Exception:
        print("[LiteMono_crf] mmcv.runner.load_checkpoint 無法匯入，略過載入。")
        return None

    try:
        ckpt = mmcv_load_checkpoint(
            model,
            filename,
            map_location=map_location,
            strict=strict,
            logger=logger,
        )
        print(f"[LiteMono_crf] 成功載入 checkpoint: {filename}")
        return ckpt
    except FileNotFoundError:
        print(f"[LiteMono_crf] 找不到 checkpoint 檔案：{filename}，略過載入。")
        return None
