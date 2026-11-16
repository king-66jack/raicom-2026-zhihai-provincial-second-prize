# -*- coding: utf-8 -*-


import os
import torch
import numpy as np
import cv2

label = ['cloudy', 'rainy', 'snowy', 'sunny']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ===== 改这两行就行 =====
ts_path = ""
img_size = 224  


model = torch.jit.load(ts_path, map_location=device)
model.eval()



def val_transform(img_rgb: np.ndarray, size: int) -> np.ndarray:
    img = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return img



def predict(X):
    if X is None:
        raise ValueError("cv2.imread 读取失败，图片为空")

    X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
    X = val_transform(X, img_size)
    X = X[np.newaxis, :, :, :]
    X = torch.from_numpy(X).float().to(device)

    with torch.no_grad():
        logits = model(X)

    pred_idx = int(torch.argmax(logits, dim=1).item())
    return label[pred_idx]



if __name__ == "__main__":
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    import argparse 
    val_dir = ""

    parser = argparse.ArgumentParser()
    parser.add_argument('--ts_path', type=str, default=ts_path)
    parser.add_argument('--img_size', type=int, default=img_size)
    parser.add_argument('--val_dir', type=str, default="/root/Image_Classify/data_cleaning/mo_mwd_combined_split/test")
    args = parser.parse_args()
    
    ts_path = args.ts_path
    img_size = args.img_size
    val_dir = args.val_dir
    
    print(f"\n>>> 本地测试模式: 重新加载模型 {ts_path} (img_size={img_size})")
    model = torch.jit.load(ts_path, map_location=device)
    model.eval()

    y_true, y_pred = [], []
    total, failed = 0, 0

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

            y_true.append(cls)
            y_pred.append(predict(img))
            total += 1

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 60)
    print("Total images:", total)
    print("Failed images:", failed)
    print("Accuracy =", acc)
    print("F1 Macro =", macro_f1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=label, target_names=label, digits=4, zero_division=0))
    print("=" * 60)