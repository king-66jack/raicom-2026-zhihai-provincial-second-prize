# -*- coding: utf-8 -*-


import os
import json
import torch
import timm


# =====================================================
# 路径配置：改这里
# =====================================================

ckpt_path = ""

save_path = ""

meta_save_path = ""


# =====================================================
# 工具函数
# =====================================================

def strip_module_prefix(state_dict):
    """
    去掉 DDP 训练可能产生的 module. 前缀。
    """
    new_state = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    return new_state


def build_model_from_cfg(cfg):
    """
    从 checkpoint 的 config 里还原模型结构。

    注意：
    - ConvNeXtV2：不主动传 img_size。
    - Swin：主动传 img_size。
    - EVA02：主动传 img_size。
    """
    model_arch = cfg.get("model_arch", None)

    if model_arch is None:
        raise RuntimeError(
            "checkpoint 的 config 里没有 model_arch，无法自动构建模型。"
            "请确认 ckpt 是训练脚本保存出来的 best_ema_model_*.pth。"
        )

    model_family = cfg.get("model_family", "")
    num_classes = int(cfg.get("num_classes", 4))
    drop_path_rate = float(cfg.get("drop_path_rate", 0.0))
    img_size = int(cfg.get("img_size", 224))

    create_kwargs = dict(
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )

    # 按你的要求：Swin / EVA02 都主动传 img_size
    if (
        model_family in ("swin", "eva02")
        or "swin" in model_arch.lower()
        or "eva02" in model_arch.lower()
    ):
        create_kwargs["img_size"] = img_size

    print("\nBuild Model:")
    print("model_arch:", model_arch)
    print("model_family:", model_family)
    print("num_classes:", num_classes)
    print("drop_path_rate:", drop_path_rate)
    print("img_size:", img_size)
    print("create_kwargs:", create_kwargs)

    model = timm.create_model(model_arch, **create_kwargs)

    return model, img_size


def save_meta_json(meta_save_path, ckpt_path, save_path, ckpt, cfg, label, img_size):
    """
    保存 TorchScript 对应的元信息。
    最终写 main.py 时，可以参考这里面的类别顺序和预处理方式。
    """
    meta = {
        "ckpt_path": ckpt_path,
        "torchscript_path": save_path,

        "checkpoint_type": ckpt.get("checkpoint_type"),
        "epoch": ckpt.get("epoch"),
        "best_f1": ckpt.get("best_f1"),
        "val_macro_f1": ckpt.get("val_macro_f1"),
        "val_acc": ckpt.get("val_acc"),

        "model_arch": cfg.get("model_arch"),
        "model_family": cfg.get("model_family"),
        "num_classes": int(cfg.get("num_classes", len(label))),
        "img_size": int(cfg.get("img_size", img_size)),
        "drop_path_rate": float(cfg.get("drop_path_rate", 0.0)),
        "dataset_classes": label,

        "preprocess": {
            "read": "cv2.imread，输入为 BGR",
            "color": "BGR -> RGB",
            "resize": "cv2.resize INTER_LINEAR",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "layout": "HWC -> CHW",
            "input_tensor_shape": [1, 3, img_size, img_size],
        },
    }

    meta_dir = os.path.dirname(meta_save_path)
    if meta_dir:
        os.makedirs(meta_dir, exist_ok=True)

    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)


# =====================================================
# 主流程
# =====================================================

def main():
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print("Loading checkpoint:")
    print(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    print("\nCheckpoint Info:")
    print("ckpt keys:", list(ckpt.keys()))
    print("checkpoint_type:", ckpt.get("checkpoint_type"))
    print("epoch:", ckpt.get("epoch"))
    print("best_f1:", ckpt.get("best_f1"))
    print("val_macro_f1:", ckpt.get("val_macro_f1"))
    print("val_acc:", ckpt.get("val_acc"))

    checkpoint_type = str(ckpt.get("checkpoint_type", ""))

    if "resume" in checkpoint_type.lower():
        print("\n⚠️ 警告：当前 checkpoint 看起来是 resume checkpoint。")
        print("如果你想复现训练验证 F1，建议使用 best_ema_model_*.pth。")

    if "model_state_dict" not in ckpt:
        raise KeyError("checkpoint 中没有 model_state_dict，无法导出。")

    cfg = ckpt.get("config", {})

    label = cfg.get("dataset_classes", ["cloudy", "rainy", "snowy", "sunny"])

    print("\nConfig From Checkpoint:")
    print("model_arch:", cfg.get("model_arch"))
    print("model_family:", cfg.get("model_family"))
    print("num_classes:", cfg.get("num_classes"))
    print("img_size:", cfg.get("img_size"))
    print("drop_path_rate:", cfg.get("drop_path_rate"))
    print("dataset_classes:", label)

    # 1. 构建模型
    model, img_size = build_model_from_cfg(cfg)

    # 2. 加载权重
    state_dict = strip_module_prefix(ckpt["model_state_dict"])

    msg = model.load_state_dict(state_dict, strict=True)

    print("\nLoad State Dict:")
    print("strict load success")
    print("missing keys:", len(msg.missing_keys), msg.missing_keys[:20])
    print("unexpected keys:", len(msg.unexpected_keys), msg.unexpected_keys[:20])

    # 3. 固定到 CPU 上导出，减少环境差异
    model.eval()
    model = model.cpu()

    example = torch.randn(1, 3, img_size, img_size)

    print("\nTracing TorchScript...")

    with torch.inference_mode():
        # 原模型输出
        origin_out = model(example)

        # trace 导出
        traced = torch.jit.trace(
            model,
            example,
            strict=False,
            check_trace=False,
        )

        traced.eval()

        # freeze 适合纯推理模型
        traced = torch.jit.freeze(traced)

        # TorchScript 输出
        traced_out = traced(example)

        max_abs_diff = (origin_out - traced_out).abs().max().item()
        mean_abs_diff = (origin_out - traced_out).abs().mean().item()

    print("\nTorchScript Check:")
    print("max_abs_diff:", max_abs_diff)
    print("mean_abs_diff:", mean_abs_diff)

    if max_abs_diff > 1e-4:
        print("⚠️ 警告：TorchScript 输出和原模型输出差异偏大，请检查导出结果。")
    else:
        print("✅ TorchScript 输出检查通过。")

    # 4. 保存 TorchScript
    traced.save(save_path)

    # 5. 保存 meta 信息
    save_meta_json(
        meta_save_path=meta_save_path,
        ckpt_path=ckpt_path,
        save_path=save_path,
        ckpt=ckpt,
        cfg=cfg,
        label=label,
        img_size=img_size,
    )

    print("\nSaved:")
    print("TorchScript:", save_path)
    print("Meta JSON:", meta_save_path)
    print("=" * 80)


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=ckpt_path)
    parser.add_argument('--save_path', type=str, default=save_path)
    parser.add_argument('--meta_save_path', type=str, default=meta_save_path)
    args = parser.parse_args()
    
    # 覆盖全局变量
    
    ckpt_path = args.ckpt_path
    save_path = args.save_path
    meta_save_path = args.meta_save_path

    main()