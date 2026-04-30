# predict_bw.py
import argparse
import os
import torch
import torch.nn.functional as F

# --------------------------
# 1) 默认的均值/方差与尺寸
# --------------------------
NORMS = {
    "cifar10":  {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616), "size": 32, "num_classes": 10},
    "cifar100": {"mean": (0.5071, 0.4865, 0.4409), "std": (0.2673, 0.2564, 0.2762), "size": 32, "num_classes": 100},
    "stl10":    {"mean": (0.4467, 0.4398, 0.4066), "std": (0.2241, 0.2215, 0.2239), "size": 96, "num_classes": 10},
    # 小ImageNet给一组常见ImageNet归一化，尺寸可通过 --img-size 指定
    "smallimagenet": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225), "size": 32, "num_classes": 127},
}

def build_model(arch: str, dataset: str):
    """构建与你训练一致的模型结构。"""
    cfg = NORMS[dataset]
    num_classes = cfg["num_classes"]

    if arch == "wideresnet":
        import models.wideresnet as models
        # 训练脚本里WRN默认 depth=28, width=2（见你的训练代码）
        model = models.build_wideresnet(depth=28, widen_factor=2, dropout=0, num_classes=num_classes)
    elif arch == "resnet":
        import models.resnet_ori as models
        model = models.ResNet50(num_classes=num_classes, rotation=True, classifier_bias=True)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model

def load_checkpoint(model, ckpt_path: str, use_ema: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if use_ema:
        if "ema_state_dict" not in ckpt or ckpt["ema_state_dict"] is None:
            raise ValueError("Checkpoint 中未保存 ema_state_dict，请去掉 --use-ema 或使用包含 EMA 的权重。")
        model.load_state_dict(ckpt["ema_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    return ckpt

def make_bw_batch(dataset: str, img_size: int = None, device="cpu"):
    cfg = NORMS[dataset]
    C = 3
    H = W = img_size or cfg["size"]
    mean = torch.tensor(cfg["mean"], dtype=torch.float32, device=device).view(1, C, 1, 1)
    std  = torch.tensor(cfg["std"],  dtype=torch.float32, device=device).view(1, C, 1, 1)

    # 生成黑(0)与白(1)两张图，[0,1]范围
    black = torch.zeros((1, C, H, W), dtype=torch.float32, device=device)
    white = torch.ones( (1, C, H, W), dtype=torch.float32, device=device)

    # 归一化： (x - mean) / std
    black_n = (black - mean) / std
    white_n = (white - mean) / std

    x = torch.cat([black_n, white_n], dim=0)   # shape: (2, C, H, W)
    colors = ["black", "white"]
    return x, colors

def to_prob(logits: torch.Tensor):
    return F.softmax(logits, dim=1)

def print_vector(name, vec):
    # 便于快速查看：最多打印前 20 维，维度多时省略
    K = vec.numel()
    if K <= 20:
        s = ", ".join([f"{v:.4f}" for v in vec.tolist()])
    else:
        head = ", ".join([f"{v:.4f}" for v in vec[:10].tolist()])
        tail = ", ".join([f"{v:.4f}" for v in vec[-10:].tolist()])
        s = f"{head}, ..., {tail}"
    print(f"{name} [{K}]: {s}")

def save_csv(path, records):
    import csv
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 记录为长表：每一行是 (image,color,head,type,class_0,...)
    # type ∈ {"logit","prob"}
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header_written = False
        for r in records:
            vec = r["vector"]
            row = [r["image"], r["color"], r["head"], r["type"]] + [float(x) for x in vec]
            if not header_written:
                header = ["image","color","head","type"] + [f"class_{i}" for i in range(len(vec))]
                writer.writerow(header)
                header_written = True
            writer.writerow(row)
    print(f"[INFO] 已保存到 CSV: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="训练保存的 checkpoint 路径（.pth.tar）")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10","cifar100","stl10","smallimagenet"])
    parser.add_argument("--arch", default="wideresnet", choices=["wideresnet","resnet"])
    parser.add_argument("--gpu-id", type=int, default=0, help="若指定则用对应 GPU")
    parser.add_argument("--use-ema", action="store_true", help="若 checkpoint 含有 ema_state_dict，则加载 EMA 权重")
    parser.add_argument("--img-size", type=int, default=None, help="覆盖默认尺寸；不填则按数据集默认")
    parser.add_argument("--save-csv", type=str, default=None, help="可选：将结果保存为 CSV")
    args = parser.parse_args()

    # 设备
    if args.gpu_id is not None and torch.cuda.is_available():
        device = torch.device("cuda", args.gpu_id)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # 构建与训练一致的模型
    model = build_model(args.arch, args.dataset).to(device)
    load_checkpoint(model, args.checkpoint, use_ema=args.use_ema)
    model.eval()

    # 生成纯色图像 batch
    x, colors = make_bw_batch(args.dataset, img_size=args.img_size, device=device)

    # 前向：与你训练/测试流程一致 -> model(x) 得到特征，再走两个头
    with torch.no_grad():
        feat = model(x)
        logits_head   = model.classify(feat)   # 主头
        logits_head_b = model.classify1(feat)  # 平衡头 / 辅助头

        prob_head   = to_prob(logits_head)
        prob_head_b = to_prob(logits_head_b)

    # 打印
    print("\n===== Predictions on pure BLACK / WHITE =====")
    records = []
    for i, color in enumerate(colors):
        print(f"\n--- {color.upper()} ---")
        l_main = logits_head[i].detach().cpu()
        l_bal  = logits_head_b[i].detach().cpu()
        p_main = prob_head[i].detach().cpu()
        p_bal  = prob_head_b[i].detach().cpu()

        print_vector("logits (head)", l_main)
        print_vector("prob   (head)", p_main)
        print_vector("logits (head_b)", l_bal)
        print_vector("prob   (head_b)", p_bal)

        # Top-5 索引（如果类别不足5，就取到上限）
        k = min(5, l_main.numel())
        topk_main = torch.topk(p_main, k=k).indices.tolist()
        topk_bal  = torch.topk(p_bal,  k=k).indices.tolist()
        print(f"Top-{k} (head):   {topk_main}")
        print(f"Top-{k} (head_b): {topk_bal}")

        records += [
            {"image": i, "color": color, "head": "head",   "type": "logit", "vector": l_main.tolist()},
            {"image": i, "color": color, "head": "head",   "type": "prob",  "vector": p_main.tolist()},
            {"image": i, "color": color, "head": "head_b", "type": "logit", "vector": l_bal .tolist()},
            {"image": i, "color": color, "head": "head_b", "type": "prob",  "vector": p_bal .tolist()},
        ]

    # 可选：保存 CSV
    save_csv(args.save_csv, records)

if __name__ == "__main__":
    main()
