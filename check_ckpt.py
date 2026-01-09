# tools/ckpt_probe.py
import os
import sys
import inspect
import argparse
import torch
from mmcv import Config
from models import MODELS


def build_model(cfg):
    name = cfg.model.name if hasattr(cfg.model, "name") else cfg.model["name"]
    # 專案自訂的 MODELS.build 需要名稱字串 + option/opt
    try:
        model = MODELS.build(name, option=cfg)
    except TypeError:
        try:
            model = MODELS.build(name, opt=cfg)
        except TypeError:
            # 退而求其次：直接取回類別來實例化
            cls = MODELS.get(name)
            try:
                model = cls(cfg)
            except TypeError:
                model = cls(option=cfg)
    return model


def where(obj):
    m = inspect.getmodule(obj)
    return getattr(m, "__file__", None) if m else None


def find_key(sd, prefer):
    # 在 state_dict 裡找一個最接近的 key（避免前綴差異）
    if prefer in sd:
        return prefer
    # 寬鬆: 用尾端匹配
    for k in sd.keys():
        if k.endswith(prefer.split("encoder/")[-1].replace("/", ".")):
            return k
    # 更寬鬆: 只匹配最後兩段
    tail = ".".join(prefer.split(".")[-5:])
    for k in sd.keys():
        if k.endswith(tail):
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="path to config yaml")
    ap.add_argument("--ckpt", required=False, help="path to ckpt (.ckpt/.pth)")
    ap.add_argument("--assert-tiny", action="store_true",
                    help="assert stem_out==32")
    ap.add_argument("--show-mismatch", action="store_true",
                    help="print first 40 mismatched tensors")
    args = ap.parse_args()

    cfg = Config.fromfile(args.cfg)
    print("=== NOW cfg.model (partial) ===")
    for k in ["max_disp", "use_concat_volume", "use_3d_decoder", "encoder", "eval_mode"]:
        if hasattr(cfg.model, k):
            print(f"{k} = {getattr(cfg.model, k)}")

    # 1) 建現在的模型
    model = build_model(cfg)

    # 2) 現在實作與通道
    enc = getattr(model.disp_net, "feat_encoder").encoder
    impl_file = where(enc.__class__)
    stem_out = enc.downsample_layers[0][0].conv.weight.shape[0]
    s0 = enc.stages[0][0].pwconv1.weight.shape[0]
    s1 = enc.stages[1][0].pwconv1.weight.shape[0]
    print("\n[ENC_IMPL_FILE]", impl_file)
    print("[ENC_CHANNEL_PROFILE_NOW]", {"stem": int(
        stem_out), "stage0": int(s0), "stage1": int(s1)})
    if args.assert_tiny:
        assert stem_out == 32, f"Expect tiny stem=32, got {stem_out}"

    # 3) 若提供 ckpt，反推當時的通道 & 列 mismatch
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        sd_old = ckpt.get("state_dict", ckpt)
        # 反推三個位置的通道
        keys_want = {
            "stem": "disp_net.feat_encoder.encoder.downsample_layers.0.0.conv.weight",
            "stage0_pw1": "disp_net.feat_encoder.encoder.stages.0.0.pwconv1.weight",
            "stage1_pw1": "disp_net.feat_encoder.encoder.stages.1.0.pwconv1.weight",
        }
        report = {}
        for tag, pref in keys_want.items():
            k = find_key(sd_old, pref)
            if k is None:
                report[tag] = "N/A"
            else:
                report[tag] = int(sd_old[k].shape[0])
        print("[ENC_CHANNEL_PROFILE_CKPT]", report)

        # mismatch 摘要
        sd_now = model.state_dict()
        mism = []
        for k, v in sd_old.items():
            if k in sd_now and tuple(v.shape) != tuple(sd_now[k].shape):
                mism.append((k, tuple(v.shape), tuple(sd_now[k].shape)))
        print(f"\nMismatched tensors: {len(mism)}")
        if args.show_mismatch:
            for k, s_old, s_new in mism[:40]:
                print(f"{k}: ckpt{s_old} -> now{s_new}")


if __name__ == "__main__":
    main()
