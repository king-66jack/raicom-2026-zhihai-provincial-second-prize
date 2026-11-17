# -*- coding: utf-8 -*-
"""
天气图像分类二阶段官方数据微调脚本（Stage-2 Official Fine-tune）

使用场景：
- 第一阶段：official_train + extra_train 混合训练，每类平衡到约 3000。
- 第二阶段：加载第一阶段 best_ema_model_*.pth，只用 official_train 小学习率微调。
- 验证集：始终只用 official_val，不参与训练、不参与微调。

核心原则：
1. init_checkpoint：第一阶段 best_ema_model_*.pth，只作为二阶段初始化。
2. resume_checkpoint：二阶段中断后继续训练用的 raw resume checkpoint。
3. official_val 从头到尾只用于验证和选模型，绝对不要放进 train_dir。

输出保持和第一阶段一致：
- best_ema_model_*.pth：用于推理 / 提交 / 下一阶段初始化。
- best_resume_raw_*.pth：用于从二阶段最佳 epoch 继续训练。
- last_resume_raw_*.pth：用于普通断点续训。
- metrics.json、classification_report.csv、confusion_matrix.csv、predictions.csv、wrong_samples.csv。

运行示例：
python weather_stage2_official_finetune.py \
  --experiment S2_E0_convnextv2_tiny_official_ft \
  --init_checkpoint /root/Image_Classify/output/student_convnextv2_tiny-combined2/best_ema_model_xxx.pth

torchrun --nproc_per_node=6 weather_stage2_official_finetune.py \
  --experiment S2_E2_swin_base_official_ft \
  --init_checkpoint /root/Image_Classify/output/teacher_swin_base_combined2/best_ema_model_xxx.pth
"""

import os
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import warnings
from contextlib import nullcontext
import csv
import json
import math
import random
import shutil
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets

import timm
from timm.utils import ModelEmaV2
from safetensors.torch import load_file as load_safetensors

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2


def apply_warning_filters():
    """只屏蔽当前环境中已知的非致命兼容/性能提示，不屏蔽真实报错。"""
    warnings.filterwarnings(
        "ignore",
        message="A new version of Albumentations is available.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Grad strides do not match bucket view strides.*",
        category=UserWarning,
    )


def amp_autocast(device: torch.device, enabled: bool):
    """AMP autocast 兼容封装。

    AMP autocast 兼容封装，不要求升级 PyTorch：
    - 新版优先使用 torch.amp.autocast
    - 旧版自动回退 torch.cuda.amp.autocast
    - CPU 或 disabled 时返回 nullcontext
    """
    enabled = bool(enabled) and device.type == "cuda"
    if not enabled:
        return nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type="cuda", enabled=True)
        except TypeError:
            try:
                return torch.amp.autocast("cuda", enabled=True)
            except TypeError:
                pass

    return torch.cuda.amp.autocast(enabled=True)


def make_amp_scaler(enabled: bool):
    """GradScaler 兼容封装，不要求升级 PyTorch。"""
    enabled = bool(enabled)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            try:
                return torch.amp.GradScaler(enabled=enabled)
            except TypeError:
                pass

    return torch.cuda.amp.GradScaler(enabled=enabled)


# =========================================================
# 1. 大参数表：以后 E0/E1/E2 尽量都从这里统一改
# =========================================================
PARAMS: Dict[str, Any] = {
    # 当前默认跑二阶段 student。也可以命令行 --experiment 覆盖。
    "active_experiment": "S2_E0_convnextv2_tiny_official_ft",

    # 二阶段数据路径：
    # train_dir 必须是 official_train，也就是官方数据 80%。
    # val_dir 必须是 official_val，也就是官方数据 20%。
    # 注意：extra_train 不应该出现在二阶段 train_dir 里。
    "train_dir": "/root/Image_Classify/data_cleaning/mo_cleaned_train",
    "val_dir": "/root/Image_Classify/data_cleaning/mo_cleaned_val",

    # 第一阶段 best_ema_model_*.pth。
    # 这个 checkpoint 只作为二阶段初始化，不恢复 optimizer/scheduler。
    # 可以在命令行用 --init_checkpoint 覆盖。
    "init_checkpoint": "",
    "min_init_match_ratio": 0.95,

    # 如果二阶段训练中断，用这个恢复 raw resume checkpoint。
    # 注意：resume_checkpoint 是二阶段自己的 last_resume_raw/best_resume_raw，
    # 不建议把第一阶段 best_ema_model 填到这里；第一阶段模型请填 init_checkpoint。
    "resume_checkpoint": "",

    # 数据集类别顺序。ImageFolder 会按文件夹名字排序，通常是 cloudy/rainy/snowy/sunny。
    "num_classes": 4,
    "dataset_classes": ["cloudy", "rainy", "snowy", "sunny"],

    # 输出错分文件夹时的类别别名。你想看 rain 文件夹，就把 rainy 映射成 rain。
    "class_alias_for_output": {
        "cloudy": "cloudy",
        "rainy": "rain",
        "snowy": "snowy",
        "sunny": "sunny",
    },

    # 保存路径
    "root_save_path": "/root/Image_Classify/output_stage2_official_ft",

    # 二阶段公共参数：小学习率、少 epoch、短 patience。
    "epochs": 10,
    "batch_size": 32,
    "num_workers": 8,
    "lr": 3e-5,
    "weight_decay": 1.5e-4,
    "warmup_epochs": 0,
    "min_lr": 1e-8,
    "patience": 8,
    "save_every": 2,
    "grad_clip": 1.0,
    "label_smoothing": 0.01,
    "seed": 2024,

    # 训练稳定性：保持和第一阶段一致。
    "use_amp": True,
    "use_ema": False,
    "ema_decay": 0.9992,
    "use_sync_bn": True,
    "channels_last": False,
    "suppress_known_warnings": True,

    # 输入尺寸
    "img_size": 224,

    # 轻增强参数：保持和第一阶段同口径，避免二阶段突然换增强导致比较不公平。
    "aug_horizontal_flip_p": 0.5,
    "aug_light_crop": True,
    "aug_crop_scale": [0.92, 1.00],
    "aug_crop_ratio": [0.95, 1.05],
    "aug_blur_p": 0.10,
    "aug_blur_limit": [3, 3],
    "aug_random_erasing_p": 0.10,
    "aug_erasing_hole_ratio": 0.08,

    # 二阶段默认关闭 Mixup：目标是用官方 hard label 校准第一阶段模型。
    # 如果你想和第一阶段完全一致，可以改成 True，并把 mixup_alpha 设回 0.15。
    "use_mixup": False,
    "mixup_alpha": 0.05,
    "use_cutmix": False,
    "use_randaugment": False,
    "use_color_jitter": False,
    "use_rotation": False,
    "use_vertical_flip": False,

    # 评估输出
    "save_eval_outputs": True,
    "copy_wrong_images": False,
    "save_probs_in_csv": True,

    # Stage-2 默认不强制加载原始 ImageNet 预训练权重，因为 init_checkpoint 才是核心。
    # 如果 init_checkpoint 为空，仍会尝试加载 local_weight_path，但不强制。
    "min_pretrained_match_ratio": 0.50,
}


# =========================================================
# 2. 实验表：模型差异只放在这里，数据增强不在这里改
# =========================================================
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "S2_E0_convnextv2_tiny_official_ft": {
        "stage": "S2",
        "role": "student_official_finetune",
        "model_tag": "stage2_ft_student_convnextv2_tiny_official-stage2-close-ema1",
        "model_arch": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "model_family": "convnextv2",
        "pretrained": False,
        "require_pretrained": False,
        "local_weight_path": "/root/Image_Classify/pretrain/ConvNeXt_V2_model.safetensors",
        "init_checkpoint": "/root/Image_Classify/output2/student_convnextv2_tiny-combined3/best_ema_model_20260626_171952_ep30_0.9647.pth",  # 建议填第一阶段 E0 的 best_ema_model_*.pth，或命令行传入
        "drop_path_rate": 0.15,
        "batch_size": 32,
        "lr": 1e-5,
        "img_size": 224,
    },

    "S2_E1_convnextv2_base_official_ft": {
        "stage": "S2",
        "role": "teacher_official_finetune",
        "model_tag": "stage2_ft_teacher_convnextv2_base_official",
        "model_arch": "convnextv2_base.fcmae_ft_in22k_in1k",
        "model_family": "convnextv2",
        "pretrained": False,
        "require_pretrained": False,
        "local_weight_path": "/root/Image_Classify/pretrain/timm_convnextv2_base_fcmae_ft_in22k_in1k(89M).safetensors",
        "init_checkpoint": "/root/Image_Classify/output2/student_convnextv2_tiny-combined3/best_ema_model_20260626_171952_ep30_0.9647.pth",  # 建议填第一阶段 E1 的 best_ema_model_*.pth，或命令行传入
        "drop_path_rate": 0.15,
        "batch_size": 16,
        "lr": 2e-5,
        "img_size": 224,
    },

    "S2_E2_swin_base_official_ft": {
        "stage": "S2",
        "role": "teacher_official_finetune",
        "model_tag": "stage2_ft_teacher_swin_base_official",
        "model_arch": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        "model_family": "swin",
        "pretrained": False,
        "require_pretrained": False,
        "local_weight_path": "/root/Image_Classify/pretrain/swin_base_patch4_window7_224.ms_in22k_ft_in1k.safetensors",
        "init_checkpoint": "/root/Image_Classify/new_upload_pth-pt_2/best_ema_model_20260626_172957_ep24_0.9755_swin_base-combined3-best.pth",  # 建议填第一阶段 E2 的 best_ema_model_*.pth，或命令行传入
        "drop_path_rate": 0.18,
        "batch_size": 16,
        "lr": 1.5e-5,
        "img_size": 224,
    },

    "S2_E3_eva02_base_mim_official_ft": {
        "stage": "S2",
        "role": "teacher_official_finetune",
        "model_tag": "stage2_ft_teacher_eva02_base_mim_official+close_ema",
        "model_arch": "eva02_base_patch14_224.mim_in22k",
        "model_family": "eva02",
        "pretrained": False,
        "require_pretrained": False,
        "local_weight_path": "/root/Image_Classify/pretrain/eva02_base_patch14_224.mim_in22k.safetensors",
        "init_checkpoint": "/root/Image_Classify/new_upload_pth-pt_2/best_ema_model_20260626_174617_ep20_0.9648_EVA-combined3.pth",  # 建议填第一阶段 E3 的 best_ema_model_*.pth，或命令行传入
        "drop_path_rate": 0.10,
        "batch_size": 16,
        "lr": 8e-5,
        "img_size": 224,
    },
}
# 8e-6

# =========================================================
# 3. 基础工具函数
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None, choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--root_save_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--local_weight_path", type=str, default=None)
    parser.add_argument("--init_checkpoint", type=str, default=None, help="第一阶段 best_ema_model_*.pth，只作为二阶段初始化。")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="二阶段 raw resume checkpoint，用于断点续训。")
    parser.add_argument("--allow_missing_pretrained", action="store_true", help="允许本地预训练权重不存在时继续训练；不建议用于 E0/E1/E2 正式实验。")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(PARAMS)
    exp_name = args.experiment or cfg["active_experiment"]
    cfg.update(EXPERIMENTS[exp_name])
    cfg["active_experiment"] = exp_name

    for key in [
        "train_dir", "val_dir", "root_save_path", "epochs", "batch_size", "lr",
        "img_size", "local_weight_path", "init_checkpoint", "resume_checkpoint",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    if getattr(args, "allow_missing_pretrained", False):
        cfg["require_pretrained"] = False

    return cfg


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    return rank, world_size, local_rank, device


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_mean_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def broadcast_tensor_from_rank0(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.broadcast(tensor, src=0)
    return tensor


def safe_json_dump(obj: Any, path: str) -> None:
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, list):
            return [convert(v) for v in x]
        return x

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(obj), f, indent=4, ensure_ascii=False)


def mixup_data(x, y, alpha, device):
    """Mixup: linearly mix two images and their labels."""
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =========================================================
# 4. 数据集与增强：所有模型共用这里
# =========================================================
class AlbumentationsImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"读取图片失败: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, target, path


def make_random_resized_crop(img_size: int, scale: List[float], ratio: List[float]):
    # 兼容 Albumentations 1.x 和 2.x
    try:
        return A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=tuple(scale),
            ratio=tuple(ratio),
            p=1.0,
        )
    except TypeError:
        return A.RandomResizedCrop(
            img_size,
            img_size,
            scale=tuple(scale),
            ratio=tuple(ratio),
            p=1.0,
        )


def make_light_erasing(cfg: Dict[str, Any]):
    """
    轻微 random erasing。
    这里按 Albumentations 1.x 的参数写法优先，避免 1.4.18 下 fill 被忽略的 warning。
    若未来换到 Albumentations 2.x，再自动回退到新版参数。
    """
    img_size = int(cfg["img_size"])
    hole = max(4, int(img_size * float(cfg["aug_erasing_hole_ratio"])))
    p = float(cfg["aug_random_erasing_p"])

    try:
        # Albumentations 1.x / 1.4.18 兼容写法
        return A.CoarseDropout(
            max_holes=1,
            min_holes=1,
            max_height=hole,
            max_width=hole,
            min_height=hole,
            min_width=hole,
            fill_value=0,
            p=p,
        )
    except TypeError:
        # Albumentations 2.x 写法
        return A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(hole, hole),
            hole_width_range=(hole, hole),
            fill=0,
            p=p,
        )


def build_transforms(cfg: Dict[str, Any]):
    img_size = int(cfg["img_size"])

    train_ops = []
    if cfg["aug_light_crop"]:
        train_ops.append(
            make_random_resized_crop(
                img_size,
                cfg["aug_crop_scale"],
                cfg["aug_crop_ratio"],
            )
        )
    else:
        train_ops.append(A.Resize(img_size, img_size))
        
        
    train_ops.extend([
        A.HorizontalFlip(p=float(cfg["aug_horizontal_flip_p"])),
        A.GaussianBlur(
            blur_limit=tuple(cfg["aug_blur_limit"]),
            p=float(cfg["aug_blur_p"]),
        ),
        make_light_erasing(cfg),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # ======== 新增：模拟线上脏样本（雾、雨、暗光、噪点、重度模糊）=======
    # train_ops.extend([
    #     # 暗光、过曝、对比度偏移
    #     A.RandomBrightnessContrast(
    #         brightness_limit=(-0.4, 0.2),
    #         contrast_limit=0.3,
    #         p=0.4
    #     ),
    #     # 镜头低光照噪点
    #     A.GaussNoise(var_limit=(10, 80), p=0.3),
    #     # 大范围模糊：水雾、远距离失焦、玻璃雾
    #     A.GaussianBlur(blur_limit=(3, 11), p=0.3),
    #     # 全局雾模拟
    #     A.RandomFog(fog_coef_lower=0.15, fog_coef_upper=0.65, p=0.3),
    #     # 雨痕遮挡画面
    #     A.RandomRain(slant_lower=-12, slant_upper=12, drop_length=8, p=0.25),
    # ])
    # # ==================================================================

    # train_ops.extend([
    #     A.HorizontalFlip(p=float(cfg["aug_horizontal_flip_p"])),
    #     # 注意：原来单独一行 A.GaussianBlur 已经删掉，上面新增段统一处理模糊
    #     make_light_erasing(cfg),
    #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ToTensorV2(),
    # ])
    

    val_ops = [
        # 天气分类更依赖全局信息，验证阶段直接 Resize，避免 center crop 丢掉天空/地面线索。
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

    return A.Compose(train_ops), A.Compose(val_ops)


def build_dataloaders(cfg: Dict[str, Any], rank: int, world_size: int):
    train_transform, val_transform = build_transforms(cfg)
    train_dataset = AlbumentationsImageFolder(cfg["train_dir"], transform=train_transform)
    val_dataset = AlbumentationsImageFolder(cfg["val_dir"], transform=val_transform)

    expected = cfg["dataset_classes"]
    if train_dataset.classes != expected:
        print(f"⚠️ 训练集类别顺序与 PARAMS['dataset_classes'] 不一致")
        print(f"   ImageFolder: {train_dataset.classes}")
        print(f"   PARAMS:      {expected}")
        print("   后续指标将以 ImageFolder 实际类别顺序为准。")
        cfg["dataset_classes"] = train_dataset.classes

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=(int(cfg["num_workers"]) > 0),
    )

    # 验证只在 rank0 上完整跑一遍，不用 DistributedSampler，避免指标和错分文件不完整。
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
        drop_last=False,
        persistent_workers=(int(cfg["num_workers"]) > 0),
    )

    return train_dataset, val_dataset, train_loader, val_loader, train_sampler


# =========================================================
# 5. 模型构建：ConvNeXt / Swin 都走 timm，但 arch 不同
# =========================================================
def load_weight_file(path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    """
    兼容常见本地权重格式：
    1) .safetensors: safetensors.torch.load_file
    2) .pth / .pt / .bin: torch.load
    3) torch checkpoint 外层可能是 model / state_dict / model_state_dict / module
    """
    if not path:
        return {}

    lower = path.lower()
    if lower.endswith(".safetensors"):
        return load_safetensors(path, device="cpu")

    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "module"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def strip_known_prefixes(key: str) -> str:
    """反复去掉常见 wrapper 前缀，兼容 DDP / 自定义 backbone wrapper / timm checkpoint。"""
    prefixes = ["module.", "model.", "backbone.", "student.", "teacher."]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """
    返回多个候选版本，尽量兼容 module./backbone./model. 前缀。
    保留 raw 版本，同时生成 strip_known_prefixes 后的版本。
    """
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = [("raw", state_dict)]
    stripped = {strip_known_prefixes(k): v for k, v in state_dict.items()}
    if stripped.keys() != state_dict.keys():
        candidates.append(("strip_common_prefixes", stripped))

    # 兼容少数 checkpoint：键名中可能出现 features.module.xxx 这类局部 module 前缀。
    local_module_stripped = {k.replace(".module.", "."): v for k, v in stripped.items()}
    if local_module_stripped.keys() != stripped.keys():
        candidates.append(("strip_inner_module", local_module_stripped))

    return candidates


def load_pretrained_safely(
    model: nn.Module,
    weight_path: str,
    rank: int,
    require_pretrained: bool = False,
    min_match_ratio: float = 0.50,
) -> None:
    """
    安全加载本地预训练权重。

    关键原则：
    - ConvNeXtV2 的 .safetensors 和 Swin 的 .pth/.pt 都支持。
    - 只加载 key 存在且 shape 完全一致的参数，自动跳过分类头 shape mismatch。
    - require_pretrained=True 时，如果路径为空、不存在、完全不匹配或匹配比例过低，直接报错。
      这样可以避免 teacher 实际随机初始化但你没发现。
    """
    if not weight_path:
        msg = "local_weight_path 为空，不加载本地预训练权重。"
        if require_pretrained:
            raise FileNotFoundError("require_pretrained=True，但 " + msg)
        if rank == 0:
            print("ℹ️ " + msg)
        return

    if not os.path.exists(weight_path):
        msg = f"本地权重不存在: {weight_path}"
        if require_pretrained:
            raise FileNotFoundError(msg)
        if rank == 0:
            print(f"⚠️ {msg}，跳过加载。")
        return

    if rank == 0:
        print(f"✅ 尝试加载本地权重: {weight_path}")

    raw_sd = load_weight_file(weight_path)
    if not isinstance(raw_sd, dict):
        raise TypeError(f"权重文件解析结果不是 dict，无法加载: {weight_path}")

    model_sd = model.state_dict()
    best_matched: Dict[str, torch.Tensor] = {}
    best_skipped_shape: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    best_name = "raw"

    for name, cand in normalize_state_dict_keys(raw_sd):
        matched: Dict[str, torch.Tensor] = {}
        skipped_shape: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
        for k, v in cand.items():
            if k in model_sd and hasattr(v, "shape"):
                if model_sd[k].shape == v.shape:
                    matched[k] = v
                else:
                    skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        if len(matched) > len(best_matched):
            best_matched = matched
            best_skipped_shape = skipped_shape
            best_name = name

    match_ratio = len(best_matched) / max(len(model_sd), 1)

    if len(best_matched) == 0:
        msg = "没有匹配到任何预训练权重 key，请检查 model_arch 和权重文件是否对应。"
        if require_pretrained:
            raise RuntimeError(msg + f" weight_path={weight_path}")
        if rank == 0:
            print("⚠️ " + msg)
        return

    if require_pretrained and match_ratio < float(min_match_ratio):
        raise RuntimeError(
            "预训练权重匹配比例过低，可能是 model_arch 与权重文件不匹配。"
            f" matched={len(best_matched)}/{len(model_sd)} ({match_ratio:.2%}), "
            f"candidate={best_name}, weight_path={weight_path}"
        )

    msg = model.load_state_dict(best_matched, strict=False)

    if rank == 0:
        print(f"📊 权重加载结果: {best_name}")
        print(f"   成功加载参数数: {len(best_matched)} / 当前模型参数数: {len(model_sd)} ({match_ratio:.2%})")
        print(f"   shape 不匹配并跳过: {len(best_skipped_shape)}")
        if best_skipped_shape:
            print(f"   shape mismatch 示例: {best_skipped_shape[:5]}")
        print(f"   Missing keys after partial load: {len(msg.missing_keys)}")
        print(f"   Unexpected keys after partial load: {len(msg.unexpected_keys)}")
        if len(msg.missing_keys) > 0:
            print(f"   Missing 示例: {msg.missing_keys[:8]}")
        if len(msg.unexpected_keys) > 0:
            print(f"   Unexpected 示例: {msg.unexpected_keys[:8]}")
        if require_pretrained:
            print("   ✅ require_pretrained=True，预训练权重检查通过。")


def build_model(cfg: Dict[str, Any], rank: int) -> nn.Module:
    create_kwargs = dict(
        pretrained=bool(cfg["pretrained"]),
        num_classes=int(cfg["num_classes"]),
        drop_path_rate=float(cfg["drop_path_rate"]),
    )

    if cfg.get("model_family") == "swin" or cfg.get("model_family") == "eva02":
        create_kwargs["img_size"] = int(cfg["img_size"])
        print(f"训练的尺寸：{cfg['img_size']}")

    model = timm.create_model(cfg["model_arch"], **create_kwargs)

    load_pretrained_safely(
        model,
        cfg.get("local_weight_path", ""),
        rank,
        require_pretrained=bool(cfg.get("require_pretrained", False)),
        min_match_ratio=float(cfg.get("min_pretrained_match_ratio", 0.50)),
    )
    return model



def load_stage1_init_checkpoint(cfg: Dict[str, Any], model: nn.Module, rank: int) -> bool:
    """加载第一阶段 best_ema_model_*.pth 作为二阶段初始化。

    这里只加载 model_state_dict，不加载 optimizer/scheduler/scaler。
    这样二阶段会用新的小学习率和新的 scheduler 从 epoch 0 开始微调。
    """
    path = cfg.get("init_checkpoint", "")
    if not path:
        if rank == 0:
            print("⚠️ init_checkpoint 为空：将只使用 local_weight_path/随机初始化开始二阶段。")
            print("   正式二阶段建议传入第一阶段 best_ema_model_*.pth。")
        return False

    if not os.path.exists(path):
        raise FileNotFoundError(f"init_checkpoint 不存在: {path}")

    if rank == 0:
        print(f"🚀 加载第一阶段初始化权重: {path}")

    raw_sd = load_weight_file(path, map_location="cpu")
    if not isinstance(raw_sd, dict):
        raise TypeError(f"init_checkpoint 解析结果不是 dict，无法加载: {path}")

    model_sd = model.state_dict()
    best_matched: Dict[str, torch.Tensor] = {}
    best_name = "raw"
    best_skipped_shape: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []

    for name, cand in normalize_state_dict_keys(raw_sd):
        matched: Dict[str, torch.Tensor] = {}
        skipped_shape: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
        for k, v in cand.items():
            if k in model_sd and hasattr(v, "shape"):
                if model_sd[k].shape == v.shape:
                    matched[k] = v
                else:
                    skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        if len(matched) > len(best_matched):
            best_matched = matched
            best_name = name
            best_skipped_shape = skipped_shape

    match_ratio = len(best_matched) / max(len(model_sd), 1)
    min_ratio = float(cfg.get("min_init_match_ratio", 0.95))

    if len(best_matched) == 0 or match_ratio < min_ratio:
        raise RuntimeError(
            "第一阶段初始化权重匹配比例过低，通常是 experiment/model_arch 与 checkpoint 不对应。"
            f" matched={len(best_matched)}/{len(model_sd)} ({match_ratio:.2%}), "
            f"min_required={min_ratio:.2%}, candidate={best_name}, path={path}"
        )

    msg = model.load_state_dict(best_matched, strict=False)
    if rank == 0:
        print(f"✅ 第一阶段权重加载完成: {best_name}")
        print(f"   成功加载参数数: {len(best_matched)} / 当前模型参数数: {len(model_sd)} ({match_ratio:.2%})")
        print(f"   shape 不匹配并跳过: {len(best_skipped_shape)}")
        if best_skipped_shape:
            print(f"   shape mismatch 示例: {best_skipped_shape[:5]}")
        print(f"   Missing keys after init load: {len(msg.missing_keys)}")
        print(f"   Unexpected keys after init load: {len(msg.unexpected_keys)}")
        print("   ✅ optimizer/scheduler/scaler 不恢复，二阶段将重新初始化优化器。")
    return True

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


# =========================================================
# 6. 评估输出：metrics、混淆矩阵、预测 CSV、错分文件夹
# =========================================================
def output_class_name(class_name: str, cfg: Dict[str, Any]) -> str:
    return cfg.get("class_alias_for_output", {}).get(class_name, class_name)


def save_confusion_matrix_csv(cm: np.ndarray, class_names: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + [int(x) for x in row])


def save_classification_report_csv(report: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for k, v in report.items():
            if isinstance(v, dict):
                writer.writerow([
                    k,
                    v.get("precision", ""),
                    v.get("recall", ""),
                    v.get("f1-score", ""),
                    v.get("support", ""),
                ])
            else:
                writer.writerow([k, v, "", "", ""])


def save_predictions_csv(
    rows: List[Dict[str, Any]],
    class_names: List[str],
    path: str,
    save_probs: bool,
) -> None:
    base_fields = [
        "sample_index", "rel_path", "image_name", "image_path",
        "true_label_id", "true_label", "pred_label_id", "pred_label", "correct",
    ]
    prob_fields = [f"prob_{c}" for c in class_names] if save_probs else []
    fields = base_fields + prob_fields

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def save_wrong_samples(
    rows: List[Dict[str, Any]],
    class_names: List[str],
    cfg: Dict[str, Any],
    out_dir: str,
) -> None:
    wrong_rows = [r for r in rows if not bool(r["correct"])]

    wrong_csv = os.path.join(out_dir, "wrong_samples.csv")
    save_predictions_csv(wrong_rows, class_names, wrong_csv, bool(cfg["save_probs_in_csv"]))

    wrong_root = os.path.join(out_dir, "wrong_by_true_class")
    os.makedirs(wrong_root, exist_ok=True)

    for c in class_names:
        folder = os.path.join(wrong_root, output_class_name(c, cfg))
        os.makedirs(folder, exist_ok=True)
        names_txt = os.path.join(folder, "wrong_names.txt")
        with open(names_txt, "w", encoding="utf-8") as f:
            for r in wrong_rows:
                if r["true_label"] == c:
                    f.write(
                        f"{r['image_name']}\ttrue={r['true_label']}\tpred={r['pred_label']}\trel_path={r['rel_path']}\n"
                    )
                    if bool(cfg["copy_wrong_images"]):
                        src = r["image_path"]
                        dst_name = f"pred_{r['pred_label']}__{r['image_name']}"
                        dst = os.path.join(folder, dst_name)
                        if os.path.exists(src):
                            shutil.copy2(src, dst)


def save_eval_artifacts(eval_result: Dict[str, Any], cfg: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    class_names = cfg["dataset_classes"]

    metrics_json = os.path.join(out_dir, "metrics.json")
    safe_json_dump(eval_result["metrics"], metrics_json)

    save_confusion_matrix_csv(
        eval_result["confusion_matrix"],
        class_names,
        os.path.join(out_dir, "confusion_matrix.csv"),
    )

    save_classification_report_csv(
        eval_result["classification_report"],
        os.path.join(out_dir, "classification_report.csv"),
    )

    save_predictions_csv(
        eval_result["rows"],
        class_names,
        os.path.join(out_dir, "predictions.csv"),
        bool(cfg["save_probs_in_csv"]),
    )

    save_wrong_samples(eval_result["rows"], class_names, cfg, out_dir)


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    class_names = cfg["dataset_classes"]
    num_classes = len(class_names)

    total_loss = 0.0
    total_steps = 0
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_paths: List[str] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for imgs, labels, paths in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if bool(cfg["channels_last"]):
                imgs = imgs.contiguous(memory_format=torch.channels_last)

            with amp_autocast(device, bool(cfg["use_amp"])):
                outputs = model(imgs)
                # 验证阶段必须使用真实标签计算 CE，不做 Mixup。
                # Mixup 只用于训练阶段的数据和损失构造。
                loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += float(loss.item())
            total_steps += 1
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_paths.extend(list(paths))
            all_probs.extend(probs.detach().cpu().float().numpy())

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    rows: List[Dict[str, Any]] = []
    val_root = cfg["val_dir"]
    for i, (path, true_id, pred_id, prob) in enumerate(zip(all_paths, y_true, y_pred, all_probs)):
        true_label = class_names[int(true_id)]
        pred_label = class_names[int(pred_id)]
        try:
            rel_path = os.path.relpath(path, val_root)
        except ValueError:
            rel_path = path

        row = {
            "sample_index": i,
            "rel_path": rel_path,
            "image_name": os.path.basename(path),
            "image_path": path,
            "true_label_id": int(true_id),
            "true_label": true_label,
            "pred_label_id": int(pred_id),
            "pred_label": pred_label,
            "correct": int(true_id == pred_id),
        }
        if bool(cfg["save_probs_in_csv"]):
            for j, c in enumerate(class_names):
                row[f"prob_{c}"] = float(prob[j])
        rows.append(row)

    metrics = {
        "loss": total_loss / max(total_steps, 1),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "lowest_class_f1": float(min(report[c]["f1-score"] for c in class_names)),
        "per_class": {
            c: {
                "precision": float(report[c]["precision"]),
                "recall": float(report[c]["recall"]),
                "f1": float(report[c]["f1-score"]),
                "support": int(report[c]["support"]),
            }
            for c in class_names
        },
    }

    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm,
        "rows": rows,
    }


# =========================================================
# 7. checkpoint 保存/恢复
# =========================================================
def save_inference_checkpoint(
    path: str,
    epoch: int,
    model_to_save: nn.Module,
    best_f1: float,
    val_metrics: Dict[str, Any],
    cfg: Dict[str, Any],
    checkpoint_type: str = "ema_best_for_inference",
) -> None:
    """
    只用于推理 / 验证 / 生成 teacher logits / E7 初始化。

    注意：这里不保存 optimizer/scheduler/scaler。
    原因是 best 模型通常保存 EMA 权重，而 optimizer 动量对应的是 raw model 训练轨迹。
    如果把 EMA 权重和 raw optimizer 状态放在同一个 checkpoint 里再 resume，
    会造成参数与动量状态不一致。
    """
    torch.save({
        "checkpoint_type": checkpoint_type,
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "best_f1": best_f1,
        "val_acc": val_metrics.get("accuracy"),
        "val_loss": val_metrics.get("loss"),
        "val_macro_f1": val_metrics.get("macro_f1"),
        "config": cfg,
        "usage": "inference_or_finetune_init_only_do_not_resume_optimizer",
    }, path)


def save_resume_checkpoint(
    path: str,
    epoch: int,
    raw_model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: Any,
    model_ema: Any,
    best_f1: float,
    val_metrics: Dict[str, Any],
    cfg: Dict[str, Any],
    early_stop_counter: int,
    checkpoint_type: str = "raw_resume_checkpoint",
) -> None:
    """
    只用于断点续训。

    这里保存 raw model 权重 + optimizer/scheduler/scaler + ema_state，
    保证 resume 时参数轨迹和优化器动量状态一致。
    """
    torch.save({
        "checkpoint_type": checkpoint_type,
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "ema_state_dict": model_ema.state_dict() if model_ema is not None else None,
        "best_f1": best_f1,
        "early_stop_counter": early_stop_counter,
        "val_acc": val_metrics.get("accuracy"),
        "val_loss": val_metrics.get("loss"),
        "val_macro_f1": val_metrics.get("macro_f1"),
        "config": cfg,
        "usage": "resume_training_only",
    }, path)


def sync_ema_to_current_model(model_ema: Any, model: nn.Module, rank: int) -> None:
    """把 EMA 影子模型同步成当前 raw model 参数。用于从 inference checkpoint 初始化后。"""
    if model_ema is None:
        return
    raw_sd = unwrap_model(model).state_dict()
    try:
        model_ema.module.load_state_dict(raw_sd, strict=True)
    except Exception as exc:
        if rank == 0:
            print(f"⚠️ EMA 同步到当前模型失败，将继续训练但 EMA 可能从初始权重开始: {exc}")


def resume_if_needed(
    cfg: Dict[str, Any],
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: Any,
    model_ema: Any,
    rank: int,
    device: torch.device,
) -> Tuple[int, float, int]:
    path = cfg.get("resume_checkpoint", "")
    if not path:
        return 0, -1.0, 0
    if not os.path.exists(path):
        if rank == 0:
            print(f"⚠️ resume_checkpoint 不存在，跳过: {path}")
        return 0, -1.0, 0

    ckpt = torch.load(path, map_location=device)
    checkpoint_type = ckpt.get("checkpoint_type", "legacy_unknown")
    has_optimizer = "optimizer_state_dict" in ckpt

    unwrap_model(model).load_state_dict(ckpt["model_state_dict"], strict=False)

    # 新版 raw resume checkpoint：恢复完整训练状态。
    if checkpoint_type in {"raw_resume_checkpoint", "best_raw_resume_checkpoint", "last_raw_resume_checkpoint"} and has_optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and bool(cfg["use_amp"]):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if model_ema is not None and ckpt.get("ema_state_dict") is not None:
            model_ema.load_state_dict(ckpt["ema_state_dict"])

        start_epoch = int(ckpt.get("epoch", 0))
        best_f1 = float(ckpt.get("best_f1", -1.0))
        early_stop_counter = int(ckpt.get("early_stop_counter", 0))
        if rank == 0:
            print(
                f"🔁 已按 raw resume checkpoint 恢复训练: {path} | "
                f"start_epoch={start_epoch} | best_f1={best_f1:.4f}"
            )
        return start_epoch, best_f1, early_stop_counter

    # 非 raw-resume checkpoint 或 best_ema_model：只作为初始化，不恢复 optimizer，避免 EMA 权重 + raw optimizer 动量不一致。
    sync_ema_to_current_model(model_ema, model, rank)
    if rank == 0:
        print(f"⚠️ 当前 checkpoint_type={checkpoint_type}，不作为断点续训使用。")
        print("   已仅加载 model_state_dict；optimizer/scheduler/scaler 已重新初始化。")
        print("   这适合 E7 fine-tune 初始化或推理模型初始化，不适合无缝继续原训练。")

    # 对于 best_ema_model 这类推理 checkpoint，只作为初始化使用。
    # 二阶段 best_f1 重置为 -1，确保会保存二阶段自己的 best_ema_model。
    return 0, -1.0, 0


def append_epoch_log(log_path: str, row: Dict[str, Any]) -> None:
    exists = os.path.exists(log_path)
    fields = [
        "epoch", "lr", "train_loss", "val_loss", "val_acc", "val_macro_f1",
        "val_macro_precision", "val_macro_recall", "val_lowest_class_f1",
        "best_f1", "early_stop_counter",
    ]
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


# =========================================================
# 8. 主训练流程
# =========================================================
def main() -> None:
    args = parse_args()
    cfg = merge_config(args)

    if bool(cfg.get("suppress_known_warnings", True)):
        apply_warning_filters()

    rank, world_size, local_rank, device = setup_distributed()
    seed_everything(int(cfg["seed"]), rank)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(cfg["root_save_path"], cfg["model_tag"])

    if is_main_process(rank):
        os.makedirs(model_dir, exist_ok=True)
        safe_json_dump(cfg, os.path.join(model_dir, f"config_{run_id}.json"))
        print("=" * 80)
        print(f"实验: {cfg['active_experiment']} | stage={cfg['stage']} | role={cfg['role']}")
        print(f"模型: {cfg['model_arch']}")
        print(f"保存目录: {model_dir}")
        print(f"GPU数: {world_size}")
        print("阶段: Stage-2 官方数据微调")
        print(f"初始化 checkpoint: {cfg.get('init_checkpoint', '') or '未设置'}")
        print("训练数据: official_train；验证数据: official_val；extra 数据不进入二阶段")
        print("增强: light crop + horizontal flip + light blur + light random erasing")
        print("禁用: CutMix / RandAugment / ColorJitter / rotation / vertical flip")
        print(f"Mixup: {cfg['use_mixup']} | alpha={cfg['mixup_alpha']}")
        print("=" * 80)

    train_dataset, val_dataset, train_loader, val_loader, train_sampler = build_dataloaders(cfg, rank, world_size)

    if is_main_process(rank):
        print(f"📦 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")
        print(f"📚 类别顺序: {cfg['dataset_classes']}")

    model = build_model(cfg, rank)

    # 二阶段核心：先加载第一阶段 best_ema_model 作为初始化，再创建 optimizer/EMA。
    # 这样 EMA、optimizer 都会从二阶段初始权重开始。
    load_stage1_init_checkpoint(cfg, model, rank)

    model = model.to(device)
    if bool(cfg["channels_last"]):
        model = model.to(memory_format=torch.channels_last)

    # EMA 在 DDP 之前创建。
    model_ema = ModelEmaV2(model, decay=float(cfg["ema_decay"])) if (bool(cfg["use_ema"]) and rank == 0) else None

    if world_size > 1 and bool(cfg["use_sync_bn"]) and device.type == "cuda":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["label_smoothing"])).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    warmup_epochs = int(cfg["warmup_epochs"])
    total_epochs = int(cfg["epochs"])
    if warmup_epochs > 0:
        scheduler_warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=float(cfg["min_lr"]),
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            [scheduler_warmup, scheduler_cosine],
            [warmup_epochs],
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs, 1),
            eta_min=float(cfg["min_lr"]),
        )

    scaler = make_amp_scaler(enabled=bool(cfg["use_amp"]) and device.type == "cuda")

    start_epoch, best_f1, early_stop_counter = resume_if_needed(
        cfg, model, optimizer, scheduler, scaler, model_ema, rank, device
    )

    log_csv = os.path.join(model_dir, f"epoch_log_{run_id}.csv")

    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for imgs, labels, _paths in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if bool(cfg["channels_last"]):
                imgs = imgs.contiguous(memory_format=torch.channels_last)

            # Mixup
            if bool(cfg["use_mixup"]):
                imgs, la, lb, lam = mixup_data(imgs, labels, float(cfg["mixup_alpha"]), device)
            else:
                la, lb, lam = labels, labels, 1.0

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device, bool(cfg["use_amp"])):
                outputs = model(imgs)
                if bool(cfg["use_mixup"]):
                    loss = mixup_criterion(criterion, outputs, la, lb, lam)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if float(cfg["grad_clip"]) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

            if model_ema is not None:
                model_ema.update(model)

            train_loss_sum += float(loss.item())
            train_steps += 1

        train_loss_tensor = torch.tensor(train_loss_sum / max(train_steps, 1), device=device)
        train_loss = reduce_mean_tensor(train_loss_tensor, world_size).item()

        # ================= 验证：只在 rank0 跑全量验证 =================
        val_metrics_tensor = torch.zeros(5, device=device)
        eval_result = None

        if is_main_process(rank):
            eval_model = model_ema.module if model_ema is not None else unwrap_model(model)
            eval_result = evaluate(eval_model, val_loader, criterion, cfg, device)
            m = eval_result["metrics"]
            val_metrics_tensor = torch.tensor([
                m["loss"], m["accuracy"], m["macro_f1"], m["macro_precision"], m["macro_recall"],
            ], device=device, dtype=torch.float32)

        val_metrics_tensor = broadcast_tensor_from_rank0(val_metrics_tensor, world_size)
        val_loss, val_acc, val_f1, val_precision, val_recall = [float(x) for x in val_metrics_tensor.detach().cpu().tolist()]

        if world_size > 1:
            dist.barrier()

        current_lr = optimizer.param_groups[0]["lr"]

        # ================= 日志、保存、早停 =================
        if is_main_process(rank):
            assert eval_result is not None
            lowest_class_f1 = eval_result["metrics"]["lowest_class_f1"]

            print(f"\nEpoch {epoch + 1}/{total_epochs} | LR: {current_lr:.2e}")
            print(f"TRAIN | Loss {train_loss:.4f}")
            print(
                f"VAL   | Loss {val_loss:.4f} | Acc {val_acc:.4f} | "
                f"Macro-F1 {val_f1:.4f} | Lowest-F1 {lowest_class_f1:.4f}"
            )
            print("PER-CLASS F1:")
            for c, d in eval_result["metrics"]["per_class"].items():
                print(f"  {c:>8s}: P={d['precision']:.4f} R={d['recall']:.4f} F1={d['f1']:.4f} N={d['support']}")

            improved = val_f1 > best_f1
            if improved:
                best_f1 = val_f1
                early_stop_counter = 0

                eval_dir = os.path.join(model_dir, f"best_eval_{run_id}")
                if bool(cfg["save_eval_outputs"]):
                    save_eval_artifacts(eval_result, cfg, eval_dir)

                # 1) 推理/生成 logits/E7 初始化：保存 EMA 权重，不带 optimizer
                inference_model = model_ema.module if model_ema is not None else unwrap_model(model)
                best_ema_path = os.path.join(
                    model_dir,
                    f"best_ema_model_{run_id}_ep{epoch + 1}_{val_f1:.4f}.pth",
                )
                save_inference_checkpoint(
                    best_ema_path,
                    epoch + 1,
                    inference_model,
                    best_f1,
                    eval_result["metrics"],
                    cfg,
                    checkpoint_type="ema_best_for_inference" if model_ema is not None else "raw_best_for_inference",
                )

                # 2) 断点续训：保存 raw model + optimizer/scheduler/scaler + ema_state
                best_resume_path = os.path.join(
                    model_dir,
                    f"best_resume_raw_{run_id}_ep{epoch + 1}_{val_f1:.4f}.pth",
                )
                save_resume_checkpoint(
                    best_resume_path,
                    epoch + 1,
                    unwrap_model(model),
                    optimizer,
                    scheduler,
                    scaler,
                    model_ema,
                    best_f1,
                    eval_result["metrics"],
                    cfg,
                    early_stop_counter,
                    checkpoint_type="best_raw_resume_checkpoint",
                )

                print(f"✅ 最佳推理模型已保存: {best_ema_path}")
                print(f"🔁 最佳断点续训模型已保存: {best_resume_path}")
                print(f"📁 最佳验证输出已保存: {eval_dir}")
            else:
                early_stop_counter += 1
                print(f"⚠️ 早停计数 {early_stop_counter}/{cfg['patience']}")

            if (epoch + 1) % int(cfg["save_every"]) == 0:
                last_path = os.path.join(model_dir, f"last_resume_raw_{run_id}_ep{epoch + 1}.pth")
                save_resume_checkpoint(
                    last_path,
                    epoch + 1,
                    unwrap_model(model),
                    optimizer,
                    scheduler,
                    scaler,
                    model_ema,
                    best_f1,
                    eval_result["metrics"],
                    cfg,
                    early_stop_counter,
                    checkpoint_type="last_raw_resume_checkpoint",
                )
                print(f"💾 断点续训检查点已保存: {last_path}")

            append_epoch_log(log_csv, {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "val_macro_precision": val_precision,
                "val_macro_recall": val_recall,
                "val_lowest_class_f1": lowest_class_f1,
                "best_f1": best_f1,
                "early_stop_counter": early_stop_counter,
            })

        stop_flag = torch.tensor([0], dtype=torch.int32, device=device)
        if is_main_process(rank) and early_stop_counter >= int(cfg["patience"]):
            stop_flag[0] = 1
        if world_size > 1:
            dist.broadcast(stop_flag, src=0)

        if stop_flag.item() == 1:
            if is_main_process(rank):
                print("🛑 早停触发")
            break

        scheduler.step()

    if is_main_process(rank):
        print(f"\n🏁 训练完成！Best Macro-F1: {best_f1:.4f}")
        print(f"📂 模型目录: {model_dir}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
