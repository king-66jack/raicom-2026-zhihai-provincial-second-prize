# -*- coding: utf-8 -*-


import os
import torch
import numpy as np
import cv2




label = ['cloudy', 'rainy', 'snowy', 'sunny']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)



# final_prob = (1.0 * p1 + 1.0 * p2 + 1.0 * p3) / 3.0
MODEL_CONFIGS = [
    {
        "name": "convnextv2",
        "path": "",
        "img_size": 224,
        "weight": 
        # 0.9446
    },
    {
        "name": "swin_base",
        "path": "",
        "img_size": 224,
        "weight":
        # 0.9446
    },
    {
        "name": "eva02_base",
        "path": "",
        "img_size": 224,
        "weight": 
        # 0.9467
    },
]



ensemble_models = []

for cfg in MODEL_CONFIGS:
    model_path = cfg["path"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    ensemble_models.append({
        "name": cfg["name"],
        "model": model,
        "img_size": int(cfg["img_size"]),
        "weight": float(cfg["weight"]),
    })

print("Loaded soft vote models:")
# for item in ensemble_models:
#     print(
#         f"name={item['name']}, "
#         f"img_size={item['img_size']}, "
#         f"weight={item['weight']}"
#     )




def val_transform(img_rgb: np.ndarray, size: int) -> torch.Tensor:
    img = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = img[np.newaxis, :, :, :]      # CHW -> NCHW

    img = torch.from_numpy(img).float().to(device)

    return img




def predict(X):
    if X is None:
        raise ValueError("cv2.imread 读取失败，图片为空")

    # cv2.imread 是 BGR，训练 Dataset 里也是先转 RGB
    X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)

    # 如果多个模型 img_size 相同，就只预处理一次，避免重复 resize
    input_cache = {}

    final_prob = None
    weight_sum = 0.0

    with torch.inference_mode():
        for item in ensemble_models:
            model = item["model"]
            img_size = item["img_size"]
            weight = item["weight"]

            if weight <= 0:
                continue

            if img_size not in input_cache:
                input_cache[img_size] = val_transform(X, img_size)

            inp = input_cache[img_size]

            logits = model(inp)

            # soft vote：平均概率，不直接平均 logits
            prob = torch.softmax(logits, dim=1)

            if final_prob is None:
                final_prob = prob * weight
            else:
                final_prob = final_prob + prob * weight

            weight_sum += weight

    if final_prob is None or weight_sum <= 0:
        raise RuntimeError("没有可用的 ensemble 模型，请检查 MODEL_CONFIGS 里的 weight 和 path。")

    final_prob = final_prob / weight_sum

    pred_idx = int(torch.argmax(final_prob, dim=1).item())

    return label[pred_idx]




if __name__ == "__main__":
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    import argparse
    import json 
    val_dir = "/root/Image_Classify/data_cleaning/mo_cleaned_val"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="", help='JSON配置文件路径')
    parser.add_argument('--val_dir', type=str, default="/root/Image_Classify/data_cleaning/mo_cleaned_val")
    args = parser.parse_args()
    
    
    val_dir = args.val_dir
    
    if args.config_file and os.path.exists(args.config_file):
        print(f"\n>>> 本地测试模式: 从 {args.config_file} 加载配置并重新加载模型...")
        with open(args.config_file, 'r', encoding='utf-8') as f:
            MODEL_CONFIGS = json.load(f)
            
        ensemble_models = []
        for cfg in MODEL_CONFIGS:
            m = torch.jit.load(cfg["path"], map_location=device)
            m.eval()
            ensemble_models.append({"name": cfg["name"], "model": m, "img_size": int(cfg["img_size"]), "weight": float(cfg["weight"])})

        for item in ensemble_models:
            print(
                f"name={item['name']}, "
                f"img_size={item['img_size']}, "
                f"weight={item['weight']}"
            )

    y_true, y_pred = [], []
    total, failed = 0, 0

    print("\nStart Soft Vote Inference...")
    print("val_dir:", val_dir)

    for cls in label:
        cls_dir = os.path.join(val_dir, cls)

        if not os.path.isdir(cls_dir):
            print("类别文件夹不存在:", cls_dir)
            continue

        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)

            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)

            if img is None:
                print("读取失败:", img_path)
                failed += 1
                continue

            pred = predict(img)

            y_true.append(cls)
            y_pred.append(pred)

            total += 1

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 60)
    print("Total images:", total)
    print("Failed images:", failed)
    print("Accuracy =", acc)
    print("F1 Macro =", macro_f1)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=label,
            target_names=label,
            digits=4,
            zero_division=0,
        )
    )
    print("=" * 60)