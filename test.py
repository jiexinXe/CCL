#!/usr/bin/env python3
"""
 test.py: Load a trained model (student or EMA) and evaluate on the test set.
 Generates t-SNE plot and confusion matrix.
"""
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from dataset.cifar import DATASET_GETTERS
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# ----------------------------------------
# Reproducibility
cudnn.deterministic = True
cudnn.benchmark = False

# ----------------------------------------
def get_model(arch, depth, width, num_classes):
    if arch == 'wideresnet':
        from models.wideresnet import build_wideresnet
        model = build_wideresnet(depth=depth, widen_factor=width, dropout=0, num_classes=num_classes)
    return model

# ----------------------------------------
def extract_features_and_preds(model, loader, device):
    model.eval()
    features, preds, labels = [], [], []
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            target = target.to(device)
            feat = model(inputs)
            logits = model.classify1(feat)
            pred = logits.argmax(dim=1)
            features.append(feat.cpu().numpy())
            preds.append(pred.cpu().numpy())
            labels.append(target.cpu().numpy())
    features = np.concatenate(features, axis=0)
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, preds, labels

# ----------------------------------------
def plot_tsne(features, labels, out_dir, fmt='eps'):
    # 降维
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    feats_2d = tsne.fit_transform(features)

    # 新建 figure/axes
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        feats_2d[:, 0], feats_2d[:, 1],
        c=labels, s=5, cmap='tab10'
    )

    # 去掉坐标轴和标题
    ax.set_xticks([])  # 不显示刻度
    ax.set_yticks([])
    ax.axis('off')  # 完全关闭轴线

    # 如果你不需要图例，就把下面这一行注释掉：
    # ax.legend(*scatter.legend_elements(), frameon=False)

    # 保存时紧凑画布、无边距
    fig.savefig(
        os.path.join(out_dir, f'tsne.{fmt}'),
        format=fmt,
        bbox_inches='tight',
        pad_inches=0
    )
    plt.close(fig)
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    # feats_2d = tsne.fit_transform(features)
    # plt.figure(figsize=(8,8))
    # scatter = plt.scatter(feats_2d[:,0], feats_2d[:,1], c=labels, s=5, cmap='tab10')
    # plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05,1), loc='upper left')
    # plt.title('t-SNE Plot')
    # plt.savefig(os.path.join(out_dir, 'tsne.eps'), bbox_inches='tight')
    # plt.close()

# ----------------------------------------

def plot_tsne_png(features, labels, out_dir, fmt='png'):
    # 降维
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    feats_2d = tsne.fit_transform(features)

    # 新建 figure/axes
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        feats_2d[:, 0], feats_2d[:, 1],
        c=labels, s=5, cmap='tab10'
    )

    # 去掉坐标轴和标题
    ax.set_xticks([])  # 不显示刻度
    ax.set_yticks([])
    ax.axis('off')  # 完全关闭轴线

    # 如果你不需要图例，就把下面这一行注释掉：
    # ax.legend(*scatter.legend_elements(), frameon=False)

    # 保存时紧凑画布、无边距
    fig.savefig(
        os.path.join(out_dir, f'tsne.{fmt}'),
        format=fmt,
        bbox_inches='tight',
        pad_inches=0
    )
    plt.close(fig)

def plot_confusion_matrix(cm, class_names, out_dir):
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i,j], 'd'), ha='center', va='center',
                 color='white' if cm[i,j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.eps'))
    plt.close()

def plot_confusion_matrix_png(cm, class_names, out_dir):
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i,j], 'd'), ha='center', va='center',
                 color='white' if cm[i,j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()

# ----------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Test a saved model and generate plots')
    # core arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10','cifar100','stl10','smallimagenet'])
    parser.add_argument('--arch', type=str, default='wideresnet', choices=['wideresnet','resnet'])
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out', type=str, default='result', help='output directory')
    # dataset imbalance params
    parser.add_argument('--num-max', type=int, default=None)
    parser.add_argument('--num-max-u', type=int, default=None)
    parser.add_argument('--imb-ratio-label', type=int, default=None)
    parser.add_argument('--imb-ratio-unlabel', type=int, default=None)
    parser.add_argument('--flag-reverse-LT', type=int, default=0)
    parser.add_argument('--flag-random-ulb', type=int, default=0)
    args = parser.parse_args()

    # prepare output and device
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda', args.gpu_id if torch.cuda.is_available() else 'cpu')

    # dataset settings
    if args.dataset == 'cifar10':
        num_classes, depth, width = 10, 28, 2
        args.dataset_name = 'cifar10'
    elif args.dataset == 'cifar100':
        num_classes, depth, width = 100, 28, 2
        args.dataset_name = 'cifar100'
    elif args.dataset == 'stl10':
        num_classes, depth, width = 10, 28, 2
        args.dataset_name = 'stl10'
    else:
        num_classes, depth, width = 127, 28, 2  # smallimagenet
        args.dataset_name = 'imagenet32'
    args.num_classes = num_classes

    # load test set
    _, _, test_dataset = DATASET_GETTERS[args.dataset](args, os.path.join('datasets', args.dataset_name))
    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)

    # build model for evaluation
    model = get_model(args.arch, depth, width, num_classes)
    model = model.to(device)

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    # choose EMA or student weights
    if 'ema_state_dict' in ckpt:
        raw_state = ckpt['ema_state_dict']
        print('Using EMA model weights for evaluation')
    else:
        raw_state = ckpt.get('state_dict', ckpt)
        print('Using student model weights for evaluation')
    # strip 'module.' if present
    state_dict = OrderedDict()
    for k, v in raw_state.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        state_dict[name] = v
    model.load_state_dict(state_dict)
    print(f'Loaded model from {args.checkpoint}')

    # inference
    features, preds, labels = extract_features_and_preds(model, test_loader, device)

    # t-SNE plot
    # plot_tsne(features, labels, args.out)
    # plot_tsne_png(features, labels, args.out)
    plot_tsne(features, labels, out_dir=args.out, fmt="pdf")
    # confusion matrix
    cm = confusion_matrix(labels, preds)
    class_names = [str(i) for i in range(num_classes)]
    # plot_confusion_matrix(cm, class_names, args.out)
    # plot_confusion_matrix_png(cm, class_names, args.out)
    # 假设你已有：cm (numpy数组, K×K) 与 class_names (长度K的列表)
    plot_confusion_matrix_pdf(cm, class_names, out_dir=args.out)

    # classification report
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print(report)
    with open(os.path.join(args.out, 'classification_report.txt'), 'w') as f:
        f.write(report)

def plot_confusion_matrix_pdf(cm, class_names, out_dir, divisor=1000.0, filename='confusion_matrix.pdf'):
    """
    PDF 版本：将混淆矩阵每个元素除以 divisor，并以两位小数显示
    保存到 out_dir/filename
    """
    cm_scaled = cm.astype(np.float64) / divisor

    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm_scaled, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar(im, format='%.2f')  # 色条两位小数

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    # 阈值用于切换格内文字颜色
    thresh = cm_scaled.max() / 2.0
    for i, j in np.ndindex(cm_scaled.shape):
        plt.text(j, i, f"{cm_scaled[i, j]:.2f}",
                 ha='center', va='center',
                 color='white' if cm_scaled[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, filename), format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
