import os
import time
import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224
from timm.models.vision_transformer import vit_small_patch16_224
from timm.models.vision_transformer import vit_tiny_patch16_224

import datasets
from models import *

from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.evaluator import Evaluator

from datasets.SCOODBenchmarkDataset import SCOODDataset
import torchvision
from sklearn import metrics
from numpy.linalg import norm, pinv
from sklearn.covariance import EmpiricalCovariance
from scipy.special import logsumexp, softmax
from collections import OrderedDict
from collections import Counter
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import gc

from sklearn.metrics import pairwise_distances_argmin_min


def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate(
        (np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out

def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh

def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh

def count_numbers(lst):
    counts = Counter(lst)
    for num, count in counts.items():
        print(f"Number {num} appears {count} times.")

def cal_ood_score(logits, group_slices):
    num_groups = group_slices.shape[0]

    all_group_ood_score_MOS = []

    smax = torch.nn.Softmax(dim=-1).cuda()
    for i in range(num_groups):
        group_logit = logits[:, group_slices[i][0]: group_slices[i][1]]

        group_softmax = smax(group_logit)
        group_others_score = group_softmax[:, 0]

        all_group_ood_score_MOS.append(-group_others_score)

    all_group_ood_score_MOS = torch.stack(all_group_ood_score_MOS, dim=1)
    final_max_score_MOS, _ = torch.max(all_group_ood_score_MOS, dim=1)
    return final_max_score_MOS

def create_dir(_path):
	if not os.path.exists(_path):
		os.makedirs(_path)
                
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
   
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
                not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert cfg.prec in ["fp16", "fp32", "amp"]
    if cfg.prec == "fp32" or cfg.prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


def load_vit_to_cpu(cfg):
    backbone_name = cfg.backbone
    if backbone_name == "IN21K-ViT-B/16":
        #model = vit_base_patch16_224(pretrained=True).eval()
        model = vit_small_patch16_224(pretrained=True).eval()
        #model = vit_tiny_patch16_224(pretrained=True).eval()

    assert cfg.prec in ["fp16", "fp32", "amp"]
    if cfg.prec == "fp16":
        # ViT's default precision is fp32
        model.half()
    
    return model


class Trainer:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = Evaluator(cfg, self.many_idxs, self.med_idxs, self.few_idxs)
        self._writer = None

    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution
        expand = cfg.expand

        if cfg.backbone.startswith("CLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
           

        

        if cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50", "CIFAR100_IR100"] :
            transform_train = transforms.Compose([
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif cfg.dataset in ["ImageNet_LT", "ImageNet"] :
            # mean = [0.485, 0.456, 0.406]
            # std = [0.229, 0.224, 0.225]
            transform_train = transforms.Compose([
                transforms.Resize(232),
                transforms.CenterCrop(resolution),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        transform_plain = transforms.Compose([
                transforms.Resize(232),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            
        transform_test = transforms.Compose([
                transforms.Resize(232),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])

        print("mean:", mean)
        print("std:", std)


        train_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_train)
        train_init_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_plain)
        train_test_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_test)
        test_dataset = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test)

        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.classnames = train_dataset.classnames

        if cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50"]:
            split_cls_num_list = datasets.CIFAR100_IR100(root, train=True).cls_num_list
        else:
            split_cls_num_list = self.cls_num_list
        self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(split_cls_num_list) >= 20) & (np.array(split_cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]

        if cfg.init_head == "1_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=1)
        elif cfg.init_head == "10_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=10)
        elif cfg.init_head == "100_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=100)
        else:
            init_sampler = None

        self.train_loader = DataLoader(train_dataset,
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_init_loader = DataLoader(train_init_dataset,
            batch_size=cfg.batch_size, sampler=init_sampler, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_test_loader = DataLoader(train_test_dataset,
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.test_loader = DataLoader(test_dataset,
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)
        
        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size

        print("Total training points:", sum(self.cls_num_list))
        # print(self.cls_num_list)
        
        test_batch_size = 100
        if cfg.dataset in ["ImageNet_LT", "ImageNet"]  :
            all_ood_loader = []
            self.openood = False
            if self.openood==False :
                ood_set = torchvision.datasets.ImageFolder(root="/home/bolin/long-tailed-ood-detection/store_datasets/SCOOD/data/images/texture/", transform=transform_test)
                test_ood_loader = DataLoader(ood_set,
                    batch_size=test_batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
                all_ood_loader.append(test_ood_loader)
                ood_set = torchvision.datasets.ImageFolder(root="/mnt/b41058b3-1ef7-4ecf-be66-ffa845609316/bolin/datasets/places365/", transform=transform_test)
                test_ood_loader = DataLoader(ood_set,
                    batch_size=test_batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
                all_ood_loader.append(test_ood_loader)
                ood_set = torchvision.datasets.ImageFolder(root="/mnt/b41058b3-1ef7-4ecf-be66-ffa845609316/bolin/datasets/SUN/", transform=transform_test)
                test_ood_loader = DataLoader(ood_set,
                    batch_size=test_batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
                all_ood_loader.append(test_ood_loader)
                ood_set = torchvision.datasets.ImageFolder(root="/mnt/b41058b3-1ef7-4ecf-be66-ffa845609316/bolin/datasets/iNaturalist/", transform=transform_test)
                test_ood_loader = DataLoader(ood_set,
                    batch_size=test_batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
                all_ood_loader.append(test_ood_loader)
                ood_set = torchvision.datasets.ImageFolder(root="/mnt/b41058b3-1ef7-4ecf-be66-ffa845609316/bolin/datasets/imagenet-o/", transform=transform_test)
                test_ood_loader = DataLoader(ood_set,
                    batch_size=test_batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
                all_ood_loader.append(test_ood_loader)
            else :
                ood_set = torchvision.datasets.ImageFolder(root="/mnt/b41058b3-1ef7-4ecf-be66-ffa845609316/bolin/datasets/openoodv1-5/ninco/", transform=transform_test)
                test_ood_loader = DataLoader(ood_set,
                    batch_size=test_batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
                all_ood_loader.append(test_ood_loader)
                ood_set = torchvision.datasets.ImageFolder(root="/mnt/b41058b3-1ef7-4ecf-be66-ffa845609316/bolin/datasets/openoodv1-5/openimage_o/", transform=transform_test)
                test_ood_loader = DataLoader(ood_set,
                    batch_size=test_batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
                all_ood_loader.append(test_ood_loader)
                ood_set = torchvision.datasets.ImageFolder(root="/mnt/b41058b3-1ef7-4ecf-be66-ffa845609316/bolin/datasets/openoodv1-5/ssb_hard/", transform=transform_test)
                test_ood_loader = DataLoader(ood_set,
                    batch_size=test_batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
                all_ood_loader.append(test_ood_loader)
                ood_set = torchvision.datasets.ImageFolder(root="/mnt/b41058b3-1ef7-4ecf-be66-ffa845609316/bolin/datasets/openoodv1-5/imagenet_c/", transform=transform_test)
                test_ood_loader = DataLoader(ood_set,
                    batch_size=test_batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
                all_ood_loader.append(test_ood_loader)
                ood_set = torchvision.datasets.ImageFolder(root="/mnt/b41058b3-1ef7-4ecf-be66-ffa845609316/bolin/datasets/openoodv1-5/imagenet_es/", transform=transform_test)
                test_ood_loader = DataLoader(ood_set,
                    batch_size=test_batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
                all_ood_loader.append(test_ood_loader)
                ood_set = torchvision.datasets.ImageFolder(root="/mnt/b41058b3-1ef7-4ecf-be66-ffa845609316/bolin/datasets/openoodv1-5/imagenet_r/", transform=transform_test)
                test_ood_loader = DataLoader(ood_set,
                    batch_size=test_batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
                all_ood_loader.append(test_ood_loader)
                ood_set = torchvision.datasets.ImageFolder(root="/mnt/b41058b3-1ef7-4ecf-be66-ffa845609316/bolin/datasets/openoodv1-5/imagenet_v2/", transform=transform_test)
                test_ood_loader = DataLoader(ood_set,
                    batch_size=test_batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
                all_ood_loader.append(test_ood_loader)
                
                
            self.all_ood_loader = all_ood_loader
        elif cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50", "CIFAR100_IR100"] :
            all_ood_set = ['texture', 'svhn', 'cifar10', 'tin', 'lsun', 'places365']
            all_ood_loader = []
            for i in all_ood_set :
                ood_set = SCOODDataset(os.path.join("/home/bolin/long-tailed-ood-detection/store_datasets/", 'SCOOD'), id_name='cifar100', ood_name=i, transform=transform_test)
                ood_loader = DataLoader(ood_set, batch_size=test_batch_size, shuffle=False, num_workers=cfg.num_workers,
                                    drop_last=False, pin_memory=True)
                all_ood_loader.append(ood_loader)
            self.all_ood_loader = all_ood_loader
        elif cfg.dataset in ["CIFAR10", "CIFAR10_IR10", "CIFAR10_IR50", "CIFAR10_IR100"] :
            all_ood_set = ['texture', 'svhn', 'cifar100', 'tin', 'lsun', 'places365']
            all_ood_loader = []
            for i in all_ood_set :
                ood_set = SCOODDataset(os.path.join("/home/bolin/long-tailed-ood-detection/store_datasets/", 'SCOOD'), id_name='cifar10', ood_name=i, transform=transform_test)
                ood_loader = DataLoader(ood_set, batch_size=test_batch_size, shuffle=False, num_workers=cfg.num_workers,
                                drop_last=False, pin_memory=True)
                all_ood_loader.append(ood_loader)
            self.all_ood_loader = all_ood_loader

    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        print("Building model")
        if cfg.zero_shot:
            assert cfg.backbone.startswith("CLIP")
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg)
            self.model = ZeroShotCLIP(clip_model)
            self.model.to(self.device)
            self.tuner = None
            self.head = None

            prompts = self.get_tokenized_prompts(classnames)
            self.model.init_text_features(prompts)

        elif cfg.backbone.startswith("CLIP"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg)
            self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        elif cfg.backbone.startswith("IN21K-ViT"):
            print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg)
            self.model = PeftModelFromViT(cfg, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            self.build_optimizer()
            self.build_criterion()

            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
            elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
                self.init_head_class_mean()
            elif cfg.init_head == "linear_probe":
                self.init_head_linear_probe()
            elif cfg.init_head == "maha":
                self.init_head_maha()
            else:
                print("No initialization with head")
            
            torch.cuda.empty_cache()
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_optimizer(self):
        cfg = self.cfg

        print("Turning off gradients in the model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        print("Turning on gradients in the tuner")
        for name, param in self.tuner.named_parameters():
            param.requires_grad_(True)
        print("Turning on gradients in the head")
        for name, param in self.head.named_parameters():
            param.requires_grad_(True)

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"Total params: {total_params}")
        print(f"Tuned params: {tuned_params}")
        print(f"Head params: {head_params}")
        # for name, param in self.tuner.named_parameters():
        #     print(name, param.numel())

        # self.optim = torch.optim.SGD([{"params": self.model.parameters()},
        #                               ],
        #                               lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)

        #NOTE: only give tuner and head to the optimizer
        self.optim = torch.optim.SGD([{"params": self.tuner.parameters()},
                                      {"params": self.head.parameters()}],
                                      lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = GradScaler() if cfg.prec == "amp" else None

    def build_criterion(self):
        cfg = self.cfg
        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)
        self.cls_num_list = cls_num_list 

        if cfg.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.loss_type == "Focal": # https://arxiv.org/abs/1708.02002
            self.criterion = FocalLoss()
        elif cfg.loss_type == "LDAM": # https://arxiv.org/abs/1906.07413
            self.criterion = LDAMLoss(cls_num_list=cls_num_list, s=cfg.scale)
        elif cfg.loss_type == "CB": # https://arxiv.org/abs/1901.05555
            self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "GRW": # https://arxiv.org/abs/2103.16370
            self.criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "BS": # https://arxiv.org/abs/2007.10740
            self.criterion == BalancedSoftmaxLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LA": # https://arxiv.org/abs/2007.07314
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LADE": # https://arxiv.org/abs/2012.00321
            self.criterion = LADELoss(cls_num_list=cls_num_list)
        
        print(cls_num_list)
        
    def get_tokenized_prompts(self, classnames):
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    @torch.no_grad()
    def init_head_text_feat(self):
        cfg = self.cfg
        classnames = self.classnames

        print("Initialize head with text features")
        prompts = self.get_tokenized_prompts(classnames)
        text_features = self.model.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)

        if cfg.backbone.startswith("CLIP-ViT"):
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.head.apply_weight(text_features)

    @torch.no_grad()
    def init_head_class_mean(self):
        print("Initialize head with class means")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature, _ = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature.cpu())
            all_labels.append(label.cpu())

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx: idx+cnt].mean(dim=0, keepdim=True)
            idx += cnt
            print(i, class_means[i][:5])
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means.float(), dim=-1)

        # all_logit = all_features @ class_means.t()
        # all_energy = -torch.logsumexp(all_logit, dim=1)
        # self.m_low  = all_energy.max() + 0.01
        # self.m_high = all_energy.min()
        self.head.apply_weight(class_means.to(self.device))


    @torch.no_grad()
    def init_head_linear_probe(self):
        print("Initialize head with linear probing")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature,_ = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        #clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced").fit(all_features, all_labels)
        clf = LogisticRegression(solver="liblinear", max_iter=100, penalty="l2", class_weight="balanced").fit(all_features, all_labels)
        class_weights = torch.from_numpy(clf.coef_).to(all_features.dtype).to(self.device)
        class_weights = F.normalize(class_weights, dim=-1)

        self.head.apply_weight(class_weights)

    @torch.no_grad()
    def init_head_maha(self):
        print("Initialize head with maha")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        train_means = []
        train_feat_centered = []
        for i in tqdm(range(self.num_classes)):
            fs = all_features[all_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m.cpu().numpy())
            train_feat_centered.extend((fs - _m).cpu().numpy())

        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))

        mean = torch.from_numpy(np.array(train_means)).cuda().float()
        prec = torch.from_numpy(ec.precision_).cuda().float()

        weight_input = [mean, prec]
        self.head.apply_weight(weight_input)

    def train(self):
        cfg = self.cfg

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        maha_loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()

        cls_num_ratio = self.cls_num_list / torch.sum(self.cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)

        num_epochs = cfg.num_epochs
        
        #engloss Engv3
        if cfg.dataset in ["ImageNet_LT", "ImageNet"] :
            margin = 1
            m_low = -8
            m_high = -17
            max_range = 8     #7   #4
        elif cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50", "CIFAR100_IR100"] :
            margin = 1
            #in3=21k
            m_low = -5
            m_high = -15
            # #clip
            # m_low = -12
            # m_high = -18
            max_range = 11
        elif cfg.dataset in ["CIFAR10", "CIFAR10_IR10", "CIFAR10_IR50", "CIFAR10_IR100"] :
            max_range = 7
        #max_range = math.ceil((m_low-m_high)/(margin*2))
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            end = time.time()

            
            # margin = 1
            # if epoch_idx != 0 :
            #     self.m_low = max(m_low_list) + 0.01
            #     self.m_high = max(m_high_list)

            # max_range = math.ceil((self.m_low-self.m_high)/(margin*2))
            # print(self.m_low, self.m_high)

            m_low_list = []
            m_high_list = []
            
            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                data_time.update(time.time() - end)

                image = batch[0]
                label = batch[1]
                image = image.to(self.device)
                label = label.to(self.device)

                if cfg.prec == "amp":
                    with autocast():
                        #output = self.model(image)
                        feature, feat = self.model(image)
                        output = self.model.head(feature)
                        
                        # #engloss Engv3                        
                        # fake_output = self.model.head(feature)     
                        # Ec_in = -torch.logsumexp(fake_output/1, dim=1)
                        # m_low_list.append(Ec_in.max().detach())
                        # m_high_list.append(Ec_in.min().detach())
                        
                        
                        # #GOOD RESULT  T0
                        # max_range = 6
                        # for i in range(1,max_range):
                        #     if i == 1 :
                        #         maha_loss = torch.pow(F.relu(Ec_in-m_low+i*2*margin-margin), 2).mean() + torch.pow(F.relu(Ec_in-m_low+i*2*margin)-margin, 2).mean()
                        #     else :
                        #         maha_loss += torch.pow(F.relu(margin-F.relu(Ec_in-m_low+i*2*margin-margin)), 2).mean() + torch.pow(F.relu(Ec_in-m_low+i*2*margin)-margin, 2).mean()
                                
                        # #T1
                        # max_range = 6
                        # maha_loss=0
                        # for i in range(1,max_range):
                        #     maha_loss += torch.pow(margin-F.relu(2*margin-F.relu(Ec_in-m_low+i*2*margin)), 2).mean() 
                        
                                
                        loss = self.criterion(output, label) 
                        loss_micro = loss / self.accum_step
                        self.scaler.scale(loss_micro).backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                else:
                    feature, feat = self.model(image)
                    output = self.model.head(feature.half())
                    loss = self.criterion(output, label)
                    loss_micro = loss / self.accum_step
                    loss_micro.backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.optim.step()
                        self.optim.zero_grad()

                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    correct = pred.eq(label).float()
                    acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                for _c, _y in zip(correct, label):
                    cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
                cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                mean_acc = np.mean(np.array(cls_accs))
                many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
                med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
                few_acc = np.mean(np.array(cls_accs)[self.few_idxs])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = epoch_idx * num_batches + batch_idx
                self._writer.add_scalar("train/lr", current_lr, n_iter)
                self._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
                self._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
                self._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
                self._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
                self._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
                self._writer.add_scalar("train/many_acc", many_acc, n_iter)
                self._writer.add_scalar("train/med_acc", med_acc, n_iter)
                self._writer.add_scalar("train/few_acc", few_acc, n_iter)
                
                end = time.time()

            #####################
            #self.test()
            ####################
            
            
            self.sched.step()
            torch.cuda.empty_cache()

        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        self.save_model(cfg.output_dir)

        self.test()

        # Close writer
        self._writer.close()

    @torch.no_grad()
    def test(self, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader

        self.cls_num_list = torch.tensor(self.cls_num_list).to(self.device)
        cls_num_ratio = self.cls_num_list / torch.sum(self.cls_num_list)
        LA = torch.log(cls_num_ratio)

        if self.cfg.dataset in ["ImageNet_LT", "ImageNet"]:
            class_num = 1000
        elif self.cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50", "CIFAR100_IR100"]:
            class_num = 100
        elif self.cfg.dataset in ["CIFAR10", "CIFAR10_IR10", "CIFAR10_IR50", "CIFAR10_IR100"]:
            class_num = 10
        
        ct = time.time()
        print('time', ct)
            
        features = []
        train_labels = []
        maha_features = []
        maha_train_labels = []
        print('preprocess ID .............')
        with torch.no_grad():
            for batch in tqdm(self.train_test_loader, ascii=True):
                image = batch[0]
                label = batch[1]
                image = image.to(self.device)
                label = label.to(self.device)
                
                feature, _ = self.model(image)

                maha_features.append(feature.cpu())
                maha_train_labels.append(label.cpu())     
        feature_id_train = np.concatenate(maha_features, axis=0)
        
        # 11.25#############
        train_means = []
        zs_labels = np.concatenate(maha_train_labels, axis=0)
        for i in tqdm(range(self.num_classes)):
            fs = feature_id_train[zs_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
        w = torch.tensor(train_means)
        #################
            
        ss = StandardScaler()
        pca_estimator = PCA(feature_id_train.shape[1])
        complete_vectors_train = ss.fit_transform(feature_id_train)
        _ = pca_estimator.fit_transform(complete_vectors_train)
        
        all_features = torch.cat(maha_features, dim=0)
        all_labels = torch.cat(maha_train_labels, dim=0)
        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(self.num_classes)):
            fs = all_features[all_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m.cpu().numpy())
            train_feat_centered.extend((fs - _m).cpu().numpy())
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))
        mean = torch.from_numpy(np.array(train_means)).cuda().float()
        prec = torch.from_numpy(ec.precision_).cuda().float()
        
        ##############
        w = torch.nn.Parameter(w.cuda())
        self.model.head.weight = w
        ##############
        print('self.model.head.weight', self.model.head.weight.size())
        w = (self.model.head.weight / self.model.head.weight.norm(dim=-1, keepdim=True)).detach().cpu().numpy()
        w = w.astype(np.float32)
        b = np.ones(class_num)
        u = -np.matmul(pinv(w), b)
        DIM = 256 #512
        
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - u) #
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray(
            (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

        print('computing alpha...')
        vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
        logit_id_train =  feature_id_train @ w.T 
        alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
        print(f'{alpha=:.4f}')

        def plt_distribution(save_dir, id_scores, ood_scores, test_dataset_name, method):
            sns.set(style="white", palette="muted")
            palette = ['#A8BAE3', '#55AB83']
            sns.displot({"ID": id_scores, "OOD": ood_scores}, label="id", kind = "kde", palette=palette, fill = True, alpha = 0.8)
            plt.savefig(os.path.join(save_dir,f"{test_dataset_name}.png"), bbox_inches='tight')
            #plt.savefig(os.path.join(save_dir,f"{method}_{test_dataset_name}.png"), bbox_inches='tight')
            plt.close()
            
        def plt_true_false(save_dir, true_scores, false_scores, ood_scores, test_dataset_name, method):
            sns.set(style="white", palette="muted")
            palette = ['#A8BAE3', '#55AB83', '#E3A8BA']
            sns.displot({"True": true_scores, "False": false_scores,"OOD": ood_scores}, label="id", kind = "kde", palette=palette, fill = True, alpha = 0.8)
            #sns.displot({"Head": true_scores, "Tail": false_scores,"OOD": ood_scores}, label="id", kind = "kde", palette=palette, fill = True, alpha = 0.8)
            plt.savefig(os.path.join(save_dir,f"{test_dataset_name}.png"), bbox_inches='tight')
            #plt.savefig(os.path.join(save_dir,f"{method}_{test_dataset_name}.png"), bbox_inches='tight')
            plt.close()
        
        def softmax(x):
            exp_x = np.exp(x)
            return exp_x / exp_x.sum(axis=-1, keepdims=True)
        
        softmax_id_train = softmax(all_features.detach().cpu().numpy() @ w.T)
        pred_labels_train = np.argmax(softmax_id_train, axis=-1)
        mean_softmax_train = [softmax_id_train[pred_labels_train == i].mean(axis=0) for i in tqdm(range(class_num))]
        def kl(p, q):
            return np.sum(np.where(p != 0, p * np.log(p / q), 0))

            
        weight = 0.0
        neco_dim = 100
        plot_true_false = False
        
        para = '_KL-matching'                       # '_ReAct+Energy',  "_GradNorm", "_KL-Matching",
        for para in ['_MSP', '_MLS', '_Energy', '_Maha', "_Residual", '_Vim_dim512', '_NECO100', '_KL-matching' ] :
            print(para)
            def get_scores(image, model):
                with torch.no_grad():
                    feature, feat = model(image)
                    output = model.head(feature)
                
                feature_id_val = feature.detach().cpu().numpy()
                energy_id_val = torch.logsumexp(output, axis=-1).detach().cpu().numpy()    
                if para in ['_Vim_dim512', '_Vim_dim512_false', '_Vim_dim512_true', '_Vim_dim512_grad','_Vim_dim512_v1' ,'_Vim_dim512_v0'] :
                    weight = 1.0
                    vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
                    score_id = -vlogit_id_val*weight + energy_id_val
                elif para == '_MSP':
                    score_id = softmax(output.cpu().numpy()).max(axis=-1)
                elif para == '_MLS':
                    score_id = output.cpu().numpy().max(axis=-1)
                elif para in ['_Energy', '_Energy_false', '_Energy_true', '_Energy_tail30','_Energy_train', '_Energy_grad']:
                    score_id = energy_id_val
                elif para in ['_Maha', '_Maha_false', '_Maha_true', '_Maha_9216', '_Maha_grad', '_Maha_v1', '_Maha_v0']:
                    expanded_x = feature.unsqueeze(1).expand(feature.size(0), mean.size(0), feature.size(1))
                    delta = expanded_x - mean
                    product = torch.matmul(delta, prec) * delta
                    score_id = -(product.sum(dim=-1).min(dim=-1)).values.cpu().numpy()
                elif para == "_ReAct+Energy":    
                    thresh = 0.99
                    clip = np.quantile(feature_id_train, thresh)
                    logit_id_val_clip = np.clip(feature_id_val, a_min=None, a_max=clip) @ w.T 
                    score_id = logsumexp(logit_id_val_clip, axis=-1)
                elif para == "_Residual": 
                    score_id = -norm(np.matmul(feature_id_val - u, NS), axis=-1)
                elif para in ['_NECO100', '_NECO100_false', '_NECO100_true', '_NECO100_grad', '_NECO100_v0', '_NECO100_v1']:
                    complete_vectors_test = ss.transform(feature_id_val)
                    cls_test_reduced_all = pca_estimator.transform(complete_vectors_test)
                    score_id_maxlogit = output.detach().cpu().numpy().max(axis=-1)
                    cls_test_reduced = cls_test_reduced_all[:, :neco_dim]
                    l_ID = []
                    for i in range(cls_test_reduced.shape[0]):
                        sc_complet = np.linalg.norm((complete_vectors_test[i, :]))
                        sc = np.linalg.norm(cls_test_reduced[i, :])
                        sc_finale = sc/sc_complet
                        l_ID.append(sc_finale)
                    score_id = l_ID * score_id_maxlogit
            
                elif para == '_KL-matching' :
                    score_id = pairwise_distances_argmin_min(
                        softmax(output.cpu().numpy()), np.array(mean_softmax_train), metric=kl)[1]

                    
                self.model.eval()
                return output, score_id
            eval_method1 = 'energy'
             
            print('time', time.time() - ct)
            ct = time.time()
            
            score_list = []
            score_list_true = []
            score_list_false = []
            for batch in tqdm(data_loader, ascii=True):
            #for batch in tqdm(self.train_test_loader, ascii=True):
                image = batch[0]
                label = batch[1]

                image = image.to(self.device)
                label = label.to(self.device)

                output, scores = get_scores(image, self.model)
                if plot_true_false :
                    _, prec_loc = torch.max(output, dim=1)
                    score_list_true.append(scores[prec_loc.cpu() == label.cpu()]) 
                    score_list_false.append(scores[prec_loc.cpu() != label.cpu()])
                    # score_list_true.append(scores[label.cpu() < 70]) 
                    # score_list_false.append(scores[label.cpu() >= 70])
                score_list.append(scores)

                self.evaluator.process(output, label)

            if plot_true_false :
                true_scores = np.concatenate(score_list_true, axis=0)
                false_scores = np.concatenate(score_list_false, axis=0)
            in_scores = np.concatenate(score_list, axis=0)
            results = self.evaluator.evaluate()
            print(list(results.values())[0])
            
            print('time', time.time() - ct)
            ct = time.time()
            
            if self.cfg.dataset in ["ImageNet_LT", "ImageNet"] :

                j=0
                tdn=0
                auroc_list = []
                aupr_list = []
                fpr95_list = []
                for test_ood_loader in self.all_ood_loader:
                    tdn=tdn+1
                    ood_score_list = []
                    for batch in tqdm(test_ood_loader, ascii=True):
                        # if j == 2:
                        #     break
                        # else:
                        #     j=j+1

                        image = batch[0]
                        label = batch[1]

                        image = image.to(self.device, non_blocking=True)
                        label = label.to(self.device, non_blocking=True)

                        output, scores = get_scores(image, self.model)
                        ood_score_list.append(scores)
                    if self.openood==False:
                        if tdn ==1:
                            test_dataset_name = 'texture'
                        elif tdn ==2:
                            test_dataset_name = 'Place'
                        elif tdn ==3:
                            test_dataset_name = 'SUN'
                        elif tdn ==4:
                            test_dataset_name = 'iNat'
                        elif tdn ==5:
                            test_dataset_name = 'imagenet-o'
                    else :
                        if tdn ==1:
                            test_dataset_name = 'ninco'
                        elif tdn ==2:
                            test_dataset_name = 'openimage_o'
                        elif tdn ==3:
                            test_dataset_name = 'ssb_hard'
                        elif tdn ==4:
                            test_dataset_name = 'imagenet_c'
                        elif tdn ==5:
                            test_dataset_name = 'imagenet-es'
                        elif tdn ==6:
                            test_dataset_name = 'imagenet_r'
                        elif tdn ==7:
                            test_dataset_name = 'imagenet-v2'

                    save_dir = self.cfg.output_dir +"/"  + str(weight) + para
                    create_dir(save_dir)

                    ood_scores = np.concatenate(ood_score_list, axis=0)

                    
                    if plot_true_false :
                        print('num_true', len(true_scores))
                        print('num_false', len(false_scores))
                    else :
                        print('num_id', len(in_scores))
                    print('num_ood', len(ood_scores))
                    auroc, aupr, fpr95 = get_measures(in_scores, ood_scores)
                    auroc_list.append(auroc)
                    aupr_list.append(aupr)
                    fpr95_list.append(fpr95)
                    ood_detectoin_str = 'tune_auroc: %.4f, aupr: %.4f, fpr95: %.4f' % (auroc, aupr, fpr95)
                    print(ood_detectoin_str)
                    all_scores = np.concatenate([-in_scores, -ood_scores], axis=0)
                    all_ood_labels = np.concatenate([np.zeros((in_scores.shape[0],1)), np.ones((ood_scores.shape[0],1))], axis=0)
                    fpr, tpr, thresholds = roc_curve(all_ood_labels.ravel(), all_scores.ravel())
                    score_distribution = True

                    if score_distribution:
                        # ROC curve:
                        plt.figure()
                        lw = 2
                        plt.plot(
                            fpr,
                            tpr,
                            color="darkorange",
                            lw=lw,
                            label="ROC curve (area = %0.4f, aupr = %0.4f, fpr95 = %0.4f,)" % (auroc, aupr, fpr95) 
                        )
                        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
                        plt.axhline(y=0.95, color="red", lw=lw, linestyle="--")
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel("False Positive Rate")
                        plt.ylabel("True Positive Rate")
                        plt.grid(True)
                        plt.legend(loc="lower right")
                        plt.savefig(os.path.join(save_dir, 'ROC_'+test_dataset_name+'.png' ))
                        plt.close()
                        
                    if plot_true_false:
                        plt_true_false(save_dir, true_scores, false_scores, ood_scores, test_dataset_name, eval_method1)
                    else :
                        plt_distribution(save_dir, in_scores, ood_scores, test_dataset_name, eval_method1)
                    
                    print('time', time.time() - ct)
                    ct = time.time() 

                avg_ood_detectoin_str = 'Average_tune : auroc: %.4f, aupr: %.4f, fpr95: %.4f' % (sum(auroc_list) / len(auroc_list), sum(aupr_list) / len(aupr_list), sum(fpr95_list) / len(fpr95_list))
                print(avg_ood_detectoin_str)
                if score_distribution:
                        # ROC curve:
                    plt.figure()
                    lw = 2
                    plt.plot(
                        fpr,
                        tpr,
                        color="darkorange",
                        lw=lw,
                        label="ROC curve (area = %0.4f, aupr = %0.4f, fpr95 = %0.4f,)" % (sum(auroc_list) / len(auroc_list), sum(aupr_list) / len(aupr_list), sum(fpr95_list) / len(fpr95_list)) 
                    )
                    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
                    plt.axhline(y=0.95, color="red", lw=lw, linestyle="--")
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.grid(True)
                    plt.legend(loc="lower right")
                    plt.savefig(os.path.join(save_dir, 'ROC.png' ))
                    plt.close()

                
                
            else :
                auroc_list = []
                aupr_list = []
                fpr95_list = []
                tdn=0
                for test_ood_loader in self.all_ood_loader:
                    ood_score_list = []
                    ood_score_list_untune = []
                    j=0
                    tdn = tdn+1

                    
                    for batch in tqdm(test_ood_loader, ascii=True):
                        # j += 1
                        # if j == 3:
                        #     break
                        
                        image = batch[0]
                        label = batch[1]

                        image = image.to(self.device, non_blocking=True)
                        label = label.to(self.device, non_blocking=True)

                        output, scores = get_scores(image, self.model)
                        ood_score_list.append(scores)

                    if tdn ==1:
                        test_dataset_name = 'texture'
                    elif tdn ==2:
                        test_dataset_name = 'svhn'
                    elif tdn ==3:
                        test_dataset_name = 'cifar'
                    elif tdn ==4:
                        test_dataset_name = 'tin'
                    elif tdn ==5:
                        test_dataset_name = 'lsun'
                    elif tdn ==6:
                        test_dataset_name = 'places365'
                    save_dir = self.cfg.output_dir +"/" + str(weight) + para 
                    create_dir(save_dir)

                    ood_scores = np.concatenate(ood_score_list, axis=0)

                    if plot_true_false :
                        print('num_true', len(true_scores))
                        print('num_false', len(false_scores))
                    else :
                        print('num_id', len(in_scores))
                    print('score_ood', len(ood_scores))
                    auroc, aupr, fpr95 = get_measures(in_scores, ood_scores)
                    auroc_list.append(auroc)
                    aupr_list.append(aupr)
                    fpr95_list.append(fpr95)
                    ood_detectoin_str = 'tune_auroc: %.4f, aupr: %.4f, fpr95: %.4f' % (auroc, aupr, fpr95)
                    print(ood_detectoin_str)
                    all_scores = np.concatenate([-in_scores, -ood_scores], axis=0)
                    all_ood_labels = np.concatenate([np.zeros((in_scores.shape[0],1)), np.ones((ood_scores.shape[0],1))], axis=0)
                    fpr, tpr, thresholds = roc_curve(all_ood_labels.ravel(), all_scores.ravel())
                    score_distribution = True

                    if score_distribution:
                        # ROC curve:
                        plt.figure()
                        lw = 2
                        plt.plot(
                            fpr,
                            tpr,
                            color="darkorange",
                            lw=lw,
                            label="ROC curve (area = %0.4f, aupr = %0.4f, fpr95 = %0.4f,)" % (auroc, aupr, fpr95) 
                        )
                        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
                        plt.axhline(y=0.95, color="red", lw=lw, linestyle="--")
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel("False Positive Rate")
                        plt.ylabel("True Positive Rate")
                        plt.grid(True)
                        plt.legend(loc="lower right")
                        plt.savefig(os.path.join(save_dir, 'ROC_'+test_dataset_name+'.png' ))
                        plt.close()

                        if plot_true_false:
                            plt_true_false(save_dir, true_scores, false_scores, ood_scores, test_dataset_name, eval_method1)
                        else :
                            plt_distribution(save_dir, in_scores, ood_scores, test_dataset_name, eval_method1)

                avg_ood_detectoin_str = 'Average_tune : auroc: %.4f, aupr: %.4f, fpr95: %.4f' % (sum(auroc_list) / len(auroc_list), sum(aupr_list) / len(aupr_list), sum(fpr95_list) / len(fpr95_list))
                print(avg_ood_detectoin_str)
                if score_distribution:
                        # ROC curve:
                    plt.figure()
                    lw = 2
                    plt.plot(
                        fpr,
                        tpr,
                        color="darkorange",
                        lw=lw,
                        label="ROC curve (area = %0.4f, aupr = %0.4f, fpr95 = %0.4f,)" % (sum(auroc_list) / len(auroc_list), sum(aupr_list) / len(aupr_list), sum(fpr95_list) / len(fpr95_list)) 
                    )
                    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
                    plt.axhline(y=0.95, color="red", lw=lw, linestyle="--")
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.grid(True)
                    plt.legend(loc="lower right")
                    plt.savefig(os.path.join(save_dir, 'ROC.png' ))
                    plt.close()

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return list(results.values())[0]

    def save_model(self, directory):
        tuner_dict = self.tuner.state_dict()
        head_dict = self.head.state_dict()
        checkpoint = {
            "tuner": tuner_dict,
            "head": head_dict
        }

        # remove 'module.' in state_dict's keys
        for key in ["tuner", "head"]:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        # save model
        save_path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device)
        tuner_dict = checkpoint["tuner"]
        head_dict = checkpoint["head"]

        print("Loading weights to from {}".format(load_path))
        self.tuner.load_state_dict(tuner_dict)
        self.head.load_state_dict(head_dict)