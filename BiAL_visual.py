# This code is constructed based on Pytorch Implementation of FixMatch(https://github.com/kekmodel/FixMatch-pytorch)
import argparse
import logging
import math
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy
from utils import Logger
from progress.bar import Bar
import loss.semiConLoss as scl

# === NEW: logging utils for experiment data ===
import csv, json
from pathlib import Path


def infer_scenario(args):
    # heuristic naming for plots
    if args.imb_ratio_unlabel == args.imb_ratio_label and args.flag_reverse_LT == 0:
        return "consistent"
    if args.imb_ratio_unlabel == 1 and args.flag_reverse_LT == 0:
        return "uniform"
    if args.flag_reverse_LT == 1:
        return "reverse"
    return "random"


def softmax_entropy(logits: torch.Tensor):
    p = torch.softmax(logits.float(), dim=-1)
    return float(-(p * (p.clamp_min(1e-12).log())).sum().item())


class RunRecorder:
    def __init__(self, out_dir: str, num_classes: int, enable: bool = True):
        self.enable = enable
        self.out = Path(out_dir)
        (self.out / "vectors").mkdir(parents=True, exist_ok=True)
        (self.out / "matrices").mkdir(parents=True, exist_ok=True)
        (self.out / "diagnostics").mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out / "metrics.csv"
        if self.enable and not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch", "seed", "scenario",
                    "bias_beta_eff", "b_theta_entropy", "b_theta_l2",
                    "pl_accept_rate", "pl_precision_known",
                    "tail_accept_rate_known", "tail_precision_known",
                    "energy_mean", "energy_std", "energy_min", "energy_max",
                    "log_mismatch_loginf",
                    "loss_all", "loss_cls", "loss_con", "loss_con2",
                    "test_top1_b", "test_top1_co", "test_loss",
                    "batch_time_avg"
                ])

    def log_epoch_row(self, **kw):
        if not self.enable: return
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                kw.get("epoch"), kw.get("seed"), kw.get("scenario"),
                kw.get("bias_beta_eff"), kw.get("b_theta_entropy"), kw.get("b_theta_l2"),
                kw.get("pl_accept_rate"), kw.get("pl_precision_known"),
                kw.get("tail_accept_rate_known"), kw.get("tail_precision_known"),
                kw.get("energy_mean"), kw.get("energy_std"),
                kw.get("energy_min"), kw.get("energy_max"),
                kw.get("log_mismatch_loginf"),
                kw.get("loss_all"), kw.get("loss_cls"), kw.get("loss_con"), kw.get("loss_con2"),
                kw.get("test_top1_b"), kw.get("test_top1_co"), kw.get("test_loss"),
                kw.get("batch_time_avg")
            ])

    def save_vector(self, name: str, epoch: int, vec: torch.Tensor):
        if not self.enable: return
        npy = self.out / "vectors" / f"{name}_epoch{epoch:03d}.npy"
        np.save(npy, vec.detach().float().cpu().numpy())

    def save_matrix(self, name: str, epoch: int, mat: torch.Tensor):
        if not self.enable: return
        npy = self.out / "matrices" / f"{name}_epoch{epoch:03d}.npy"
        np.save(npy, mat.detach().cpu().numpy())

    def save_json(self, name: str, epoch: int, payload: dict):
        if not self.enable: return
        j = self.out / "diagnostics" / f"{name}_epoch{epoch:03d}.json"
        with open(j, "w") as f:
            json.dump(payload, f, indent=2)


# === /NEW ===


logger = logging.getLogger(__name__)
best_acc = 0
best_acc_b = 0


# ---------------- Bias utilities (BiAL-style) ----------------
def _logmeanexp(x, dim=0):
    # log(mean(exp(x))) for numerical stability
    return torch.logsumexp(x, dim=dim) - math.log(x.size(dim))


@torch.no_grad()
def compute_bias_theta(model, args):
    was_training = model.training
    model.eval()
    B = args.bias_probe_batch
    C, H, W = 3, args.img_size, args.img_size
    noinfo = torch.zeros(B, C, H, W, device=args.device)

    out = model(noinfo)  # ← 只前向一次
    feat = out[0] if isinstance(out, tuple) else out  # ← 取特征
    z = model.classify(feat)
    zb = model.classify1(feat)
    z_co = 0.5 * z + 0.5 * zb
    b = _logmeanexp(z_co, dim=0)
    b = b - b.mean()  # 仅做标量均值居中

    if was_training:
        model.train()  # ← 恢复训练状态
    return b.detach()


def update_bias_theta(model, args, epoch):
    """
    以 EMA 刷新 b_theta；按 epoch/refresh 频率执行。
    """
    if (epoch >= args.bias_start_epoch) and ((epoch - args.bias_start_epoch) % args.bias_refresh_every == 0):
        b_now = compute_bias_theta(model, args)
        args.b_theta = args.bias_m * args.b_theta + (1.0 - args.bias_m) * b_now


def bias_ramp(epoch, warmup_epochs):
    if warmup_epochs <= 0:
        return 1.0
    t = max(0, min(epoch / float(warmup_epochs), 1.0))
    return t  # 线性 ramp，可按需换成 sigmoid


def apply_bias(z, args):
    """
    统一的 logits 去偏：E = z - beta_eff * b_theta
    """
    if getattr(args, 'b_theta', None) is None:
        return z
    return z - args.bias_beta_eff * args.b_theta


# ----------------------------------------------------------------


def make_imb_data(max_num, class_num, gamma, flag=1, flag_LT=0):
    mu = np.power(1 / gamma, 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))

    if flag == 0 and flag_LT == 1:
        class_num_list = list(reversed(class_num_list))
    return list(class_num_list)


def compute_adjustment_list(label_list, tro, args):
    label_freq_array = np.array(label_list)
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(args.device)
    return adjustments


def compute_py(train_loader, args):
    """compute the base probabilities"""
    label_freq = {}
    for i, (inputs, labell) in enumerate(train_loader):
        labell = labell.to(args.device)
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    label_freq_array = torch.from_numpy(label_freq_array)
    label_freq_array = label_freq_array.to(args.device)
    return label_freq_array


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar', epoch_p=1):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    seed = args.seed
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def compute_adjustment_by_py(py, tro, args):
    adjustments = torch.log(py ** tro + 1e-12)
    adjustments = adjustments.to(args.device)
    return adjustments


def sharp(a, T):
    a = a ** T
    a_sum = torch.sum(a, dim=1, keepdim=True)
    a = a / a_sum
    return a.detach()


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='1', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'stl10', 'smallimagenet'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnet'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=250000, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=500, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=1, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.90, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=0, type=int,
                        help="random seed")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    parser.add_argument('--num-max', default=500, type=int,
                        help='the max number of the labeled data')
    parser.add_argument('--num-max-u', default=4000, type=int,
                        help='the max number of the unlabeled data')
    parser.add_argument('--imb-ratio-label', default=1, type=int,
                        help='the imbalanced ratio of the labelled data')
    parser.add_argument('--imb-ratio-unlabel', default=1, type=int,
                        help='the imbalanced ratio of the unlabeled data')
    parser.add_argument('--flag-reverse-LT', default=0, type=int,
                        help='whether to reverse the distribution of the unlabeled data')
    parser.add_argument('--ema-mu', default=0.99, type=float,
                        help='mu when ema')

    parser.add_argument('--tau', default=2.0, type=float,
                        help='tau for head consistency')
    parser.add_argument('--est-epoch', default=10, type=int,
                        help='the start step to estimate the distribution')
    parser.add_argument('--img-size', default=32, type=int,
                        help='image size for small imagenet')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='ema ratio for estimating distribution of the unlabeled data')
    parser.add_argument('--beta', default=0.5, type=float,
                        help='ema ratio for estimating distribution of the all data')
    parser.add_argument('--lambda1', default=0.7, type=float,
                        help='coefficient of final loss')
    parser.add_argument('--lambda2', default=1.0, type=float,
                        help='coefficient of final loss')

    # ---- bias removal (BiAL) ----
    parser.add_argument('--bias-beta', type=float, default=1.0,
                        help='strength beta for bias subtraction E=z-beta*b_theta')
    parser.add_argument('--bias-m', type=float, default=0.9,
                        help='EMA momentum for b_theta')
    parser.add_argument('--bias-warmup-epochs', type=int, default=10,
                        help='epochs to ramp up beta from 0 to bias-beta')
    parser.add_argument('--bias-start-epoch', type=int, default=20,
                        help='epoch to start measuring bias')
    parser.add_argument('--bias-refresh-every', type=int, default=1,
                        help='refresh period (in epochs) for bias probing')
    parser.add_argument('--bias-probe-batch', type=int, default=16,
                        help='batch size for no-information probing (cLME)')

    # logging granularity
    parser.add_argument('--tail-ratio', type=float, default=0.3,
                        help='fraction of classes treated as tail by pi^L (e.g., bottom 30%)')
    parser.add_argument('--log-detail', type=int, default=1,
                        help='enable extra experiment logging (1=on, 0=off)')

    args = parser.parse_args()
    global best_acc
    global best_acc_b

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)

        elif args.arch == 'resnet':
            import models.resnet_ori as models
            model = models.ResNet50(num_classes=args.num_classes, rotation=True, classifier_bias=True)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank},"
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}", )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.dataset_name = 'cifar10'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.dataset_name = 'cifar100'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2

    elif args.dataset == 'stl10':
        args.num_classes = 10
        args.dataset_name = 'stl10'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2


    elif args.dataset == 'smallimagenet':
        args.num_classes = 127
        if args.img_size == 32:
            args.dataset_name = 'imagenet32'
        elif args.img_size == 64:
            args.dataset_name = 'imagenet64'

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, 'datasets/' + args.dataset_name)

    if args.local_rank == 0:
        torch.distributed.barrier()

    labeled_trainloader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    args.est_step = 0

    args.py_con = compute_py(labeled_trainloader, args)
    args.py_uni = torch.ones(args.num_classes) / args.num_classes
    # args.py_uni = args.py_uni.to(args.device)

    args.py_all = args.py_con
    args.py_unlabeled = args.py_uni

    class_list = []
    for i in range(args.num_classes):
        class_list.append(str(i))

    title = 'FixMatch-' + args.dataset
    args.logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
    args.logger.set_names(
        ['Top1_co acc', 'Top5_co acc', 'Best Top1_co acc', 'Top1_b acc', 'Top5_b acc', 'Best Top1_b acc'])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    # ---- init bias buffer ----
    args.b_theta = torch.zeros(args.num_classes, device=args.device)
    args.bias_beta_eff = 0.0

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.py_unlabeled = checkpoint['py_unlabeled']
        args.py_all = checkpoint['py_all']

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    # --- 初始化实验记录器（只在主进程启用，避免多卡并发写文件）---
    if args.local_rank in [-1, 0]:
        args.recorder = RunRecorder(out_dir=args.out,
                                    num_classes=args.num_classes,
                                    enable=bool(args.log_detail))
    else:
        # 给非主进程也挂一个“空记录器”，避免属性缺失
        class _Noop:
            def __getattr__(self, name):
                return lambda *a, **k: None

        args.recorder = _Noop()
    # --- /初始化记录器 ---

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()

    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)

    args.logger.close()


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    global best_acc
    global best_acc_b
    test_accs = []
    avg_time = []
    end = time.time()
    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
    logits_la_s = compute_adjustment_by_py(args.py_con, args.tau, args)
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    semiConLoss = scl.SemiConLoss(args.batch_size, args.batch_size, args.num_classes, args)
    semiConLoss2 = scl.softConLoss(args.batch_size, args.batch_size, args.num_classes, args)
    model.train()
    lbs = args.batch_size
    ubs = args.batch_size * args.mu
    py_labeled = args.py_con.to(args.device)
    py_unlabeled = args.py_uni.to(args.device)
    py_all = args.py_all.to(args.device)
    cut1 = lbs + 3 * ubs
    pro = ubs / (ubs + lbs)
    for epoch in range(args.start_epoch, args.epochs):

        # --- NEW: epoch accumulators for logging ---
        K = args.num_classes
        py_labeled = args.py_con.to(args.device)
        k_tail = max(1, int(K * args.tail_ratio))
        tail_idx = torch.topk(py_labeled, k=k_tail, largest=False).indices
        tail_mask_vec = torch.zeros(K, dtype=torch.bool, device=args.device)
        tail_mask_vec[tail_idx] = True

        acc_accept = 0.0  # accepted count
        acc_total = 0.0  # total unlabeled (w) count
        acc_known_accept = 0.0
        acc_known_correct = 0.0
        acc_tail_known_accept = 0.0
        acc_tail_known_correct = 0.0

        # energy running stats
        e_sum = 0.0;
        e_sqsum = 0.0;
        e_min = 1e9;
        e_max = -1e9;
        e_n = 0

        # M_t (accept confusion) on known-labeled items
        M_epoch = torch.zeros(K * K, dtype=torch.long, device='cpu')
        # --- /NEW ---

        # refresh bias & update beta ramp
        update_bias_theta(model, args, epoch)
        args.bias_beta_eff = args.bias_beta * bias_ramp(epoch - args.bias_start_epoch, args.bias_warmup_epochs)

        print('current epoch: ', epoch + 1)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_con = AverageMeter()
        losses_cls = AverageMeter()
        losses_con2 = AverageMeter()

        bar = Bar('Training', max=args.eval_step)

        num_unlabeled = torch.ones(args.num_classes).to(args.device)
        num_all = torch.ones(args.num_classes).to(args.device)
        for batch_idx in range(args.eval_step):
            try:
                (inputs_x, inputs_x_s, inputs_x_s1), targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                (inputs_x, inputs_x_s, inputs_x_s1), targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s, inputs_u_s1), u_real = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, inputs_u_s1), u_real = next(unlabeled_iter)
            u_real = u_real.to(args.device)
            mask_l = (u_real != -2).float().unsqueeze(1).to(args.device)
            data_time.update(time.time() - end)
            inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1, inputs_x_s, inputs_x_s1], dim=0).to(
                args.device)
            targets_x = targets_x.to(args.device)
            feat, feat_mlp, center_feat = model(inputs)
            # -----------------------------------------------------------------------------------------------------------
            logits = model.classify(feat[:cut1])
            logits_b = model.classify1(feat[:cut1])

            # ------ debias all logits: E = z - beta_eff * b_theta ------
            logits = apply_bias(logits, args)
            logits_b = apply_bias(logits_b, args)

            logits_x = logits[:lbs]
            logits_x_w, logits_x_s, logits_x_s1 = logits[lbs:].chunk(3)
            logits_x_b = logits_b[:lbs]
            # logits LA
            logits_x_b_w, logits_x_b_s, logits_x_b_s1 = logits_b[lbs:].chunk(3)
            del logits, logits_b
            l_u_s = F.cross_entropy(logits_x, targets_x, reduction='mean')
            l_b_s = F.cross_entropy(logits_x_b + logits_la_s, targets_x, reduction='mean')
            logits_la_u = (- compute_adjustment_by_py((1 - pro) * py_labeled + pro * py_all, 1.0, args) +
                           compute_adjustment_by_py(py_unlabeled, 1 + args.tau / 2, args))
            logits_co = 1 / 2 * (logits_x_w + logits_la_u) + 1 / 2 * logits_x_b_w
            energy = -torch.logsumexp((logits_co.detach()) / args.T, dim=1)
            pseudo_label_co = F.softmax((logits_co.detach()) / args.T, dim=1)
            pseudo_label_con = sharp(F.softmax((logits_co.detach()) / args.T, dim=1), 4.0)

            prob_co, targets_co = torch.max(pseudo_label_co, dim=-1)
            mask = prob_co.ge(args.threshold)
            mask = mask.float()

            # --- NEW: accumulate pseudo-label stats ---
            # mask: [ubs] for accepted; u_real: [ubs] with -2 meaning "unknown"
            with torch.no_grad():
                u_known = (u_real != -2)
                accepted = mask.bool()
                total_ubs = float(accepted.numel())
                acc_total += total_ubs
                acc_accept += float(accepted.sum().item())

                accepted_known = (accepted & u_known)
                if accepted_known.any():
                    y_true = u_real[accepted_known]
                    y_pl = targets_co[accepted_known]
                    correct = (y_true == y_pl)

                    acc_known_accept += float(accepted_known.sum().item())
                    acc_known_correct += float(correct.sum().item())

                    # tail subset judged by true label
                    is_tail_true = tail_mask_vec[y_true]
                    if is_tail_true.any():
                        acc_tail_known_accept += float(is_tail_true.sum().item())
                        acc_tail_known_correct += float(correct[is_tail_true].sum().item())

                    # build M_t by bincount: flatten (y_true, y_pl) to single index
                    K = args.num_classes
                    idx = (y_true.long().cpu() * K + y_pl.long().cpu())
                    bc = torch.bincount(idx, minlength=K * K)
                    M_epoch[:bc.numel()] += bc

                # energy stats on unlabeled 'w'
                e = energy.detach()
                e_sum += float(e.sum().item())
                e_sqsum += float((e ** 2).sum().item())
                e_min = min(e_min, float(e.min().item()))
                e_max = max(e_max, float(e.max().item()))
                e_n += e.numel()
            # --- /NEW ---

            targets_co = torch.cat([targets_co, targets_co], dim=0).to(args.device)
            logits_b_s = torch.cat([logits_x_b_s, logits_x_b_s1], dim=0).to(args.device)
            logits_la_u_b = compute_adjustment_by_py(py_all, args.tau, args)
            mask_twice = torch.cat([mask, mask], dim=0)
            l_u_b = (F.cross_entropy(logits_b_s + logits_la_u_b, targets_co,
                                     reduction='none') * mask_twice).mean()

            logits_u_s = torch.cat([logits_x_s, logits_x_s1], dim=0).to(args.device)
            l_u_u = (F.cross_entropy(logits_u_s, targets_co,
                                     reduction='none') * mask_twice).mean()

            loss_u = max(1.5, args.mu) * l_u_u + l_u_s
            loss_b = max(1.5, args.mu) * l_u_b + l_b_s
            loss_cls = loss_u + loss_b
            # ----------------------------------------------------------------------------------------------------------
            feat_mlp = feat_mlp[lbs:]
            f3, f4 = feat_mlp[ubs:3 * ubs, :].chunk(2)
            f1, f2 = feat_mlp[3 * ubs:, :].chunk(2)

            # ----------------------------------------------------------------------------------------------------------
            feat_mlp = torch.cat([center_feat, feat_mlp[3 * ubs:, :], feat_mlp[:3 * ubs, :]], dim=0)
            center_label = torch.ones(args.num_classes, args.num_classes).to(args.device)
            one_hot_targets = F.one_hot(targets_x, num_classes=args.num_classes)
            one_hot_targets = torch.cat([one_hot_targets, one_hot_targets], dim=0).to(args.device)
            label_contrac = torch.cat([center_label, one_hot_targets], dim=0).to(args.device)
            # la = compute_adjustment_by_py(py_all, 1.0, args)
            contrac_loss = semiConLoss(feat_mlp, label_contrac)

            # ----------------------------------------------------------------------------------------------------------
            maskcon = energy.le(-8.75)
            idx = torch.nonzero(maskcon).squeeze()
            f3 = torch.reshape(f3[idx, :], (-1, f1.shape[1]))
            f4 = torch.reshape(f4[idx, :], (-1, f1.shape[1]))
            pseudo_label_con = torch.reshape(pseudo_label_con[idx, :], (-1, args.num_classes))

            label_contrac = torch.cat([center_label, one_hot_targets, pseudo_label_con, pseudo_label_con], dim=0).to(
                args.device)
            feat_all = torch.cat([center_feat, f1, f2, f3, f4], dim=0)
            contrac_loss2 = semiConLoss2(label_contrac, feat_all, args.device)

            loss = args.lambda1 * loss_cls + args.lambda2 * contrac_loss + (1 - args.lambda1) * contrac_loss2

            loss.backward()
            losses.update(loss.item())
            losses_cls.update(loss_cls.item())
            losses_con.update(contrac_loss.item())
            losses_con2.update(contrac_loss2.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            mask = mask.unsqueeze(1).to(args.device)
            maskcon = maskcon.float().unsqueeze(1).to(args.device)
            num_all += torch.sum(pseudo_label_co * mask, dim=0)
            # num_unlabeled += torch.sum(pseudo_label_co * mask_l * mask, dim=0)
            # num_unlabeled += torch.sum(pseudo_label_co * mask_l * maskcon, dim=0)
            num_unlabeled += torch.sum(pseudo_label_co * maskcon, dim=0)
            # w_soft = mask.unsqueeze(1) * maskcon.unsqueeze(1).float()
            # num_unlabeled += torch.sum(pseudo_label_co * w_soft, dim=0)

            batch_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({batch}/{size}) | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                         'Loss: {loss:.4f} | Loss_cls: {loss_cls:.4f} | Loss_con: {loss_con:.4f} | Loss_con2: {loss_con2:.4f}'.format(
                batch=batch_idx + 1,
                size=args.eval_step,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                loss_cls=losses_cls.avg,
                loss_con=losses_con.avg,
                loss_con2=losses_con2.avg,
            )
            bar.next()
        bar.finish()

        if epoch > args.est_epoch:
            py_unlabeled = args.alpha * py_unlabeled + (1 - args.alpha) * num_unlabeled / sum(num_unlabeled)
            py_all = args.beta * py_all + (1 - args.beta) * num_all / sum(num_all)
        print('\n')
        print(py_unlabeled)
        print(py_all)
        avg_time.append(batch_time.avg)

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model
        test_la = - compute_adjustment_by_py(1 / 2 * py_labeled + 1 / 2 * py_all, 1.0, args)
        if args.local_rank in [-1, 0]:

            test_loss, test_acc, test_top5_acc, test_acc_b, test_top5_acc_b = test(args, test_loader,
                                                                                   test_model, epoch,
                                                                                   test_la)
            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc_b, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc_b > best_acc_b

            best_acc = max(test_acc, best_acc)
            best_acc_b = max(test_acc_b, best_acc_b)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            if (epoch + 1) % 10 == 0 or (is_best and epoch > 250):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                    'acc': test_acc,
                    'best_acc': best_acc_b,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'py_unlabeled': py_unlabeled,
                    'py_all': py_all
                }, is_best, args.out, epoch_p=epoch + 1)

            test_accs.append(test_acc_b)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc_b))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

            args.logger.append([test_acc, test_top5_acc, best_acc, test_acc_b, test_top5_acc_b, best_acc_b])

    # --- NEW: compute epoch-level stats & write files ---
    if args.local_rank in [-1, 0]:
        # bias metrics (current args.b_theta)
        b_entropy = softmax_entropy(args.b_theta)
        b_l2 = float(torch.norm(args.b_theta.detach()).item())

        # effective-prior mismatch (log-inf) w.r.t. labeled prior
        eps = 1e-12
        log_mismatch = float(torch.max(torch.abs(
            torch.log((py_all + eps).float()) - torch.log((py_labeled + eps).float())
        )).item())

        # pseudo-label acceptance / precision
        pl_accept_rate = float(acc_accept / max(acc_total, 1.0))
        pl_prec_known = float(acc_known_correct / max(acc_known_accept, 1.0))
        tail_acc_rate = float(acc_tail_known_accept / max(acc_total, 1.0))
        tail_prec_known = float(acc_tail_known_correct / max(acc_tail_known_accept, 1.0))

        # energy stats
        if e_n > 0:
            e_mean = e_sum / e_n
            e_var = max(e_sqsum / e_n - e_mean * e_mean, 0.0)
            e_std = e_var ** 0.5
        else:
            e_mean = e_std = 0.0;
            e_min = 0.0;
            e_max = 0.0

        # save vectors/matrix for this epoch
        args.recorder.save_vector("b_theta", epoch, args.b_theta)
        args.recorder.save_vector("py_unlabeled", epoch, py_unlabeled)
        args.recorder.save_vector("py_all", epoch, py_all)
        args.recorder.save_matrix("M_accept", epoch, M_epoch.view(args.num_classes, args.num_classes))

        # diagnostics (energy distribution snapshot)
        args.recorder.save_json("energy", epoch, {
            "mean": e_mean, "std": e_std, "min": e_min, "max": e_max, "n": int(e_n)
        })

        # main CSV row
        scenario = infer_scenario(args)
        args.recorder.log_epoch_row(
            epoch=epoch, seed=args.seed, scenario=scenario,
            bias_beta_eff=float(args.bias_beta_eff),
            b_theta_entropy=b_entropy, b_theta_l2=b_l2,
            pl_accept_rate=pl_accept_rate, pl_precision_known=pl_prec_known,
            tail_accept_rate_known=tail_acc_rate, tail_precision_known=tail_prec_known,
            energy_mean=e_mean, energy_std=e_std, energy_min=e_min, energy_max=e_max,
            log_mismatch_loginf=log_mismatch,
            loss_all=float(losses.avg), loss_cls=float(losses_cls.avg),
            loss_con=float(losses_con.avg), loss_con2=float(losses_con2.avg),
            test_top1_b=float(test_acc_b), test_top1_co=float(test_acc), test_loss=float(test_loss),
            batch_time_avg=float(batch_time.avg)
        )
    # --- /NEW ---

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch, la):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_b = AverageMeter()
    top5_b = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            # outputs_feat = model(inputs)
            # outputs = model.classify(outputs_feat)
            # outputs_b = model.classify1(outputs_feat)
            # outputs_co = 1 / 2 * (outputs + la) + 1 / 2 * outputs_b
            # loss = F.cross_entropy(outputs_b, targets)

            outputs_feat = model(inputs)
            outputs = model.classify(outputs_feat)
            outputs_b = model.classify1(outputs_feat)

            # debias at inference for train-test consistency
            outputs = apply_bias(outputs, args)
            outputs_b = apply_bias(outputs_b, args)

            outputs_co = 0.5 * (outputs + la) + 0.5 * outputs_b
            loss = F.cross_entropy(outputs_b, targets)

            prec1_b, prec5_b = accuracy(outputs_b, targets, topk=(1, 5))
            prec1_co, prec5_co = accuracy(outputs_co, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1_co.item(), inputs.shape[0])
            top5.update(prec5_co.item(), inputs.shape[0])
            top1_b.update(prec1_b.item(), inputs.shape[0])
            top5_b.update(prec5_b.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

    logger.info("top-1 acc: {:.2f}".format(top1_b.avg))
    logger.info("top-5 acc: {:.2f}".format(top5_b.avg))

    return losses.avg, top1.avg, top5.avg, top1_b.avg, top5_b.avg


if __name__ == '__main__':
    main()
