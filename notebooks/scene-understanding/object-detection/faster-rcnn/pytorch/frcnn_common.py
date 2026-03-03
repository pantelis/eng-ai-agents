"""Shared Faster R-CNN components for the 6-notebook tutorial series.

This module extracts the 13 classes and functions that were duplicated
across notebooks 01-06.  Each notebook now imports what it needs rather
than re-defining everything from scratch.

Source of truth: notebook 05 (training), which has the most recent
implementations with Copilot-generated docstrings.
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from datasets import load_dataset
from torch.utils.checkpoint import checkpoint as grad_ckpt
from torch.utils.data import IterableDataset

# ─── Constants ─────────────────────────────────────────────────────────────────

IMG_SIZE = 400  # 400x400 — fits in 16 GB VRAM with AMP + frozen backbone
NUM_CLASSES = 81  # 80 COCO + background
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# COCO 80-class names (1-indexed; 0 = background)
COCO_NAMES = [
    "__background__",
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


# ─── Utilities ─────────────────────────────────────────────────────────────────


def box_iou(boxes_a, boxes_b):
    """Compute pairwise IoU: (N,4) x (M,4) -> (N,M)."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    ix1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    iy1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    ix2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    iy2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def encode_boxes(proposals, gt_boxes):
    """Parameterize GT boxes as (tx, ty, tw, th) deltas w.r.t. proposals."""
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]
    px = proposals[:, 0] + 0.5 * pw
    py = proposals[:, 1] + 0.5 * ph
    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]
    gx = gt_boxes[:, 0] + 0.5 * gw
    gy = gt_boxes[:, 1] + 0.5 * gh
    return torch.stack(
        [(gx - px) / pw, (gy - py) / ph, torch.log(gw / pw), torch.log(gh / ph)],
        dim=1,
    )


def decode_boxes(anchors, deltas):
    """Inverse of encode_boxes with additional clamping for numerical stability."""
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    ax = anchors[:, 0] + 0.5 * aw
    ay = anchors[:, 1] + 0.5 * ah
    dx, dy, dw, dh = (
        deltas[:, 0],
        deltas[:, 1],
        deltas[:, 2].clamp(max=4.0),
        deltas[:, 3].clamp(max=4.0),
    )
    px = dx * aw + ax
    py = dy * ah + ay
    pw = torch.exp(dw) * aw
    ph = torch.exp(dh) * ah
    return torch.stack([px - 0.5 * pw, py - 0.5 * ph, px + 0.5 * pw, py + 0.5 * ph], dim=1)


# ─── Data pipeline ─────────────────────────────────────────────────────────────


class COCOStreamDataset(IterableDataset):
    """Stream COCO from HuggingFace and resize to IMG_SIZE x IMG_SIZE."""

    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        """Initialize the streaming COCO dataset reader and optional sample cap for quick experiments."""
        super().__init__()
        self.ds = load_dataset(
            "detection-datasets/coco", split=split, streaming=True
        )
        self.max_samples = max_samples

    def __iter__(self):
        """Yield normalized images and valid detection targets, skipping samples without boxes."""
        count = 0
        for sample in self.ds:
            if self.max_samples and count >= self.max_samples:
                break
            img = sample["image"].convert("RGB")
            W0, H0 = img.size
            img = img.resize((IMG_SIZE, IMG_SIZE))
            t = TF.to_tensor(img)
            t = (t - IMAGENET_MEAN) / IMAGENET_STD

            sx, sy = IMG_SIZE / W0, IMG_SIZE / H0
            boxes, labels = [], []
            for ann, cat in zip(
                sample["objects"]["bbox"], sample["objects"]["category"]
            ):
                x, y, w, h = ann
                x1, y1 = x * sx, y * sy
                x2, y2 = (x + w) * sx, (y + h) * sy
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cat) + 1)  # 0 = background

            if not boxes:
                continue

            count += 1
            yield t, {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
            }


def frcnn_collate_fn(batch):
    """Stack images but keep target dicts in a Python list for variable lengths."""
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


# ─── Backbone: ResNet50 + FPN ─────────────────────────────────────────────────


class Bottleneck(nn.Module):
    """ResNet bottleneck residual block used to build deep feature stages."""

    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        """Build the bottleneck block layers and optional projection for residual matching."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * 4)
        self.downsample = downsample

    def forward(self, x):
        """Run the residual branch and shortcut branch, then fuse them with a ReLU."""
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(x)
        return F.relu(out + identity)


class ResNet50(nn.Module):
    """Minimal ResNet-50 backbone that returns multi-scale feature maps for FPN."""

    def __init__(self):
        """Construct the ResNet stem and four backbone stages used for detection features."""
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        """Create one ResNet stage with downsampling in the first block when needed."""
        ds = None
        if stride != 1 or in_ch != out_ch * 4:
            ds = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * 4),
            )
        layers = [Bottleneck(in_ch, out_ch, stride, ds)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_ch * 4, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Compute C2-C5 feature maps; checkpoint deeper stages to reduce memory usage."""
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = grad_ckpt(self.layer3, c3, use_reentrant=False)
        c5 = grad_ckpt(self.layer4, c4, use_reentrant=False)
        return c2, c3, c4, c5


class FPN(nn.Module):
    """Feature Pyramid Network that fuses backbone stages into semantically rich pyramid levels."""

    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256):
        """Create lateral and output convolutions plus a pooled P6 pyramid level."""
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels]
        )
        self.output = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels]
        )
        self.p6 = nn.MaxPool2d(1, stride=2)

    def forward(self, features):
        """Fuse features in a top-down pathway and return pyramid levels P2-P6."""
        c2, c3, c4, c5 = features
        p5 = self.lateral[3](c5)
        p4 = self.lateral[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        outs = [self.output[i](p) for i, p in enumerate([p2, p3, p4, p5])]
        outs.append(self.p6(outs[-1]))
        return outs  # [P2, P3, P4, P5, P6]


# ─── RPN ───────────────────────────────────────────────────────────────────────


class AnchorGenerator(nn.Module):
    """Generate multi-scale, multi-aspect anchors over each FPN level."""

    def __init__(
        self,
        anchor_sizes=(32, 64, 128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        strides=(4, 8, 16, 32, 64),
    ):
        """Store anchor scales, aspect ratios, and per-level strides for anchor tiling."""
        super().__init__()
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides

    def _base(self, size):
        """Create canonical anchors centered at the origin for one scale and all ratios."""
        return torch.tensor(
            [
                [
                    -size * (r**0.5) / 2,
                    -size / (r**0.5) / 2,
                    size * (r**0.5) / 2,
                    size / (r**0.5) / 2,
                ]
                for r in self.aspect_ratios
            ],
            dtype=torch.float32,
        )

    def forward(self, feature_maps, image_size):
        """Tile base anchors across every pyramid grid location and concatenate them."""
        all_anchors = []
        for fm, sz, st in zip(feature_maps, self.anchor_sizes, self.strides):
            _, _, fh, fw = fm.shape
            base = self._base(sz)
            sx = (torch.arange(fw, device=fm.device) + 0.5) * st
            sy = (torch.arange(fh, device=fm.device) + 0.5) * st
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            shifts = torch.stack([sx, sy, sx, sy], dim=-1).reshape(-1, 4)
            all_anchors.append(
                (shifts[:, None, :] + base.to(fm.device)[None, :, :]).reshape(-1, 4)
            )
        return torch.cat(all_anchors, dim=0)


class RPNHead(nn.Module):
    """Shared convolutional RPN head that predicts objectness logits and box deltas."""

    def __init__(self, in_ch=256, k=3):
        """Initialize shared RPN conv and prediction heads for objectness and box regression."""
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.cls = nn.Conv2d(in_ch, k, 1)
        self.box = nn.Conv2d(in_ch, k * 4, 1)
        for layer in [self.conv, self.cls, self.box]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        """Apply the shared head independently to each FPN level."""
        cls_outs, box_outs = [], []
        for f in features:
            t = F.relu(self.conv(f))
            cls_outs.append(self.cls(t))
            box_outs.append(self.box(t))
        return cls_outs, box_outs


class RegionProposalNetwork(nn.Module):
    """Train/eval RPN module that scores anchors and emits final proposals."""

    RPN_BATCH = 256
    POS_FRAC = 0.5
    POS_THR = 0.7
    NEG_THR = 0.3

    def __init__(
        self, head, anchor_gen, pre_nms=2000, post_nms=1000, nms_thr=0.7, min_sz=16
    ):
        """Configure proposal filtering thresholds and connect the RPN head and anchor generator."""
        super().__init__()
        self.head = head
        self.anchor_gen = anchor_gen
        self.pre_nms = pre_nms
        self.post_nms = post_nms
        self.nms_thr = nms_thr
        self.min_sz = min_sz

    def _filter(self, props, scores, img_size):
        """Clip, size-filter, top-k rank, and NMS proposals to keep high-quality candidates."""
        H, W = img_size
        props[:, [0, 2]] = props[:, [0, 2]].clamp(0, W)
        props[:, [1, 3]] = props[:, [1, 3]].clamp(0, H)
        keep = (props[:, 2] - props[:, 0] >= self.min_sz) & (
            props[:, 3] - props[:, 1] >= self.min_sz
        )
        props, scores = props[keep], scores[keep]
        scores, order = scores.topk(min(self.pre_nms, len(scores)))
        props = props[order]
        keep = self._nms(props, scores, self.nms_thr)[: self.post_nms]
        return props[keep], scores[keep]

    @staticmethod
    def _nms(boxes, scores, thr):
        """Perform non-maximum suppression in pure PyTorch and return kept indices."""
        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
            iou = inter / (areas[i] + areas[order[1:]] - inter).clamp(1e-6)
            order = order[1:][iou <= thr]
        return torch.tensor(keep, dtype=torch.long)

    def _assign(self, anchors, gt_boxes):
        """Match anchors to GT boxes and subsample labels for balanced RPN training."""
        if gt_boxes.numel() == 0:
            return (
                torch.full(
                    (len(anchors),), -1, dtype=torch.long, device=anchors.device
                ),
                torch.zeros_like(anchors),
            )
        iou = box_iou(anchors, gt_boxes)
        max_iou, gi = iou.max(1)
        labels = torch.full(
            (len(anchors),), -1, dtype=torch.long, device=anchors.device
        )
        labels[max_iou >= self.POS_THR] = 1
        labels[max_iou < self.NEG_THR] = 0
        labels[iou.argmax(0)] = 1
        n_pos = int(self.RPN_BATCH * self.POS_FRAC)
        for val, n in [(1, n_pos), (0, self.RPN_BATCH - n_pos)]:
            idx = (labels == val).nonzero(as_tuple=False).squeeze(1)
            if len(idx) > n:
                labels[idx[torch.randperm(len(idx))[n:]]] = -1
        return labels, gt_boxes[gi]

    def forward(self, features, image_size, targets=None):
        """Produce proposals and, during training, compute RPN classification/regression losses."""
        cls_outs, box_outs = self.head(features)
        anchors = self.anchor_gen(features, image_size)
        all_scores = torch.cat(
            [c.permute(0, 2, 3, 1).reshape(c.shape[0], -1) for c in cls_outs], 1
        )
        all_deltas = torch.cat(
            [b.permute(0, 2, 3, 1).reshape(b.shape[0], -1, 4) for b in box_outs], 1
        )
        props_list = []
        for i in range(all_scores.shape[0]):
            sc = all_scores[i].sigmoid()
            pr = decode_boxes(anchors, all_deltas[i])
            pr, _ = self._filter(pr.detach(), sc.detach(), image_size)
            props_list.append(pr)
        losses = {}
        if targets is not None and self.training:
            B = all_scores.shape[0]
            c_tot = b_tot = 0.0
            for i in range(B):
                gt = targets[i]["boxes"].to(anchors.device)
                lbl, mgt = self._assign(anchors, gt)
                sam = lbl >= 0
                c_tot += F.binary_cross_entropy_with_logits(
                    all_scores[i][sam], lbl[sam].float()
                )
                pos = lbl == 1
                if pos.any():
                    b_tot += F.smooth_l1_loss(
                        all_deltas[i][pos],
                        encode_boxes(anchors[pos], mgt[pos]),
                        beta=1.0 / 9,
                    )
            losses = {"rpn_cls": c_tot / B, "rpn_box": b_tot / B}
        return props_list, losses


# ─── ROI Head ──────────────────────────────────────────────────────────────────


class ROIAlign(nn.Module):
    """Assign proposals to the right FPN level and bilinearly sample fixed grids."""

    def __init__(self, out_size=7, k0=4, k_min=2, k_max=5):
        """Set ROI output resolution and FPN level-selection parameters."""
        super().__init__()
        self.out_size = out_size
        self.k0 = k0
        self.k_min = k_min
        self.k_max = k_max

    def _level(self, boxes):
        """Map each ROI to an FPN level based on its scale."""
        areas = (
            ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
            .clamp(1e-6)
            .sqrt()
        )
        return (
            torch.floor(self.k0 + torch.log2(areas / 224.0))
            .long()
            .clamp(self.k_min, self.k_max)
            - self.k_min
        )

    def forward(self, fmaps, proposals, image_size):
        """Pool each proposal from its assigned pyramid level into fixed-size feature maps."""
        H, W = image_size
        all_feats = []
        for bi, props in enumerate(proposals):
            if len(props) == 0:
                continue
            levels = self._level(props)
            feats = torch.zeros(
                len(props),
                fmaps[0].shape[1],
                self.out_size,
                self.out_size,
                device=props.device,
            )
            for lvl, fm in enumerate(fmaps[:4]):
                mask = levels == lvl
                if not mask.any():
                    continue
                lp = props[mask]
                n = len(lp)
                x1 = lp[:, 0] / W * 2 - 1
                y1 = lp[:, 1] / H * 2 - 1
                x2 = lp[:, 2] / W * 2 - 1
                y2 = lp[:, 3] / H * 2 - 1
                gx = torch.linspace(0, 1, self.out_size, device=props.device)
                gy = torch.linspace(0, 1, self.out_size, device=props.device)
                gy_g, gx_g = torch.meshgrid(gy, gx, indexing="ij")
                gx_g = x1[:, None, None] + (x2 - x1)[:, None, None] * gx_g[None]
                gy_g = y1[:, None, None] + (y2 - y1)[:, None, None] * gy_g[None]
                grid = torch.stack([gx_g, gy_g], dim=-1)
                crops = F.grid_sample(
                    fm[bi : bi + 1].expand(n, -1, -1, -1),
                    grid,
                    align_corners=True,
                    mode="bilinear",
                    padding_mode="border",
                )
                feats[mask] = crops
            all_feats.append(feats)
        if not all_feats:
            return torch.zeros(
                0,
                fmaps[0].shape[1],
                self.out_size,
                self.out_size,
                device=fmaps[0].device,
            )
        return torch.cat(all_feats, 0)


class TwoMLPHead(nn.Module):
    """Two-layer MLP head used by Fast R-CNN for ROI feature encoding."""

    def __init__(self, in_channels=256 * 7 * 7, fc_dim=1024):
        """Create the two fully connected layers of the Fast R-CNN box head."""
        super().__init__()
        self.fc1 = nn.Linear(in_channels, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)

    def forward(self, x):
        """Flatten pooled ROI features and project them to a shared embedding."""
        return F.relu(self.fc2(F.relu(self.fc1(x.flatten(1)))))


class FastRCNNPredictor(nn.Module):
    """Final Fast R-CNN predictor that outputs class logits and class-specific box deltas."""

    def __init__(self, in_channels=1024, num_classes=81):
        """Initialize classification and box-regression linear prediction layers."""
        super().__init__()
        self.cls = nn.Linear(in_channels, num_classes)
        self.box = nn.Linear(in_channels, num_classes * 4)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.zeros_(self.cls.bias)
        nn.init.normal_(self.box.weight, std=0.001)
        nn.init.zeros_(self.box.bias)

    def forward(self, x):
        """Return class logits and class-specific box deltas for each ROI embedding."""
        return self.cls(x), self.box(x)


# ─── Full model ────────────────────────────────────────────────────────────────


class FasterRCNN(nn.Module):
    """End-to-end Faster R-CNN model combining backbone, RPN, ROI heads, and postprocessing."""

    ROI_BATCH = 512
    ROI_POS_FRAC = 0.25
    ROI_POS_THR = 0.5
    SCORE_THR = 0.05
    NMS_THR = 0.5
    MAX_DETS = 100

    def __init__(self, num_classes=81):
        """Assemble backbone, FPN, RPN, ROI modules, and freeze early backbone stages."""
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNet50()
        self.fpn = FPN()
        self.rpn = RegionProposalNetwork(RPNHead(), AnchorGenerator())
        self.roi_align = ROIAlign(out_size=7)
        self.box_head = TwoMLPHead()
        self.predictor = FastRCNNPredictor(num_classes=num_classes)

        # Freeze stem + layer1-3 to save VRAM; layer4+FPN+heads are trained
        for p in (
            list(self.backbone.stem.parameters())
            + list(self.backbone.layer1.parameters())
            + list(self.backbone.layer2.parameters())
            + list(self.backbone.layer3.parameters())
        ):
            p.requires_grad_(False)

    def _sample_rois(self, proposals, targets):
        """Build a minibatch of 512 ROIs per image with 25% positives."""
        s_props, s_labels, s_gt = [], [], []
        for props, tgt in zip(proposals, targets):
            gt_boxes = tgt["boxes"]
            gt_labels = tgt["labels"]
            all_props = torch.cat([props, gt_boxes]) if len(props) else gt_boxes
            if len(gt_boxes) == 0:
                n = min(self.ROI_BATCH, len(all_props))
                idx = torch.randperm(len(all_props))[:n]
                s_props.append(all_props[idx])
                s_labels.append(
                    torch.zeros(n, dtype=torch.long, device=props.device)
                )
                s_gt.append(all_props[idx])
                continue
            iou = box_iou(all_props, gt_boxes)
            max_iou, gi = iou.max(1)
            labels = torch.zeros(len(all_props), dtype=torch.long, device=props.device)
            pos = max_iou >= self.ROI_POS_THR
            labels[pos] = gt_labels[gi[pos]]
            n_pos = int(self.ROI_BATCH * self.ROI_POS_FRAC)
            n_neg = self.ROI_BATCH - n_pos
            pos_idx = pos.nonzero(as_tuple=False).squeeze(1)
            neg_idx = (~pos).nonzero(as_tuple=False).squeeze(1)
            pos_idx = pos_idx[torch.randperm(len(pos_idx))[:n_pos]]
            neg_idx = neg_idx[torch.randperm(len(neg_idx))[:n_neg]]
            sel = torch.cat([pos_idx, neg_idx])
            s_props.append(all_props[sel])
            s_labels.append(labels[sel])
            s_gt.append(gt_boxes[gi[sel]])
        return s_props, s_labels, s_gt

    def _roi_loss(self, cls_logits, bbox_preds, labels_list, gt_list, props_list):
        """Compute Fast R-CNN classification and box regression losses for sampled ROIs."""
        all_labels = torch.cat(labels_list)
        all_gt = torch.cat(gt_list)
        all_props = torch.cat(props_list)
        cls_loss = F.cross_entropy(cls_logits, all_labels)
        pos = all_labels > 0
        if pos.any():
            tgt_deltas = encode_boxes(all_props[pos], all_gt[pos])
            C = self.num_classes
            pred_deltas = bbox_preds[pos].view(-1, C, 4)[
                torch.arange(pos.sum()), all_labels[pos]
            ]
            box_loss = F.smooth_l1_loss(pred_deltas, tgt_deltas, beta=1.0 / 9)
        else:
            box_loss = bbox_preds.sum() * 0.0
        return cls_loss, box_loss

    def _postprocess(self, cls_logits, bbox_preds, proposals_list, image_size):
        """Decode predictions, run per-class score thresholding/NMS, and build final detections."""
        H, W = image_size
        C = self.num_classes
        results = []
        offset = 0
        for props in proposals_list:
            n = len(props)
            if n == 0:
                results.append(
                    {
                        "boxes": torch.zeros(0, 4),
                        "scores": torch.zeros(0),
                        "labels": torch.zeros(0, dtype=torch.long),
                    }
                )
                continue
            logits = cls_logits[offset : offset + n]
            deltas = bbox_preds[offset : offset + n]
            offset += n
            scores = F.softmax(logits, -1)
            all_b, all_s, all_l = [], [], []
            for ci in range(1, C):
                boxes = decode_boxes(props, deltas.view(n, C, 4)[:, ci, :])
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, W)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, H)
                sc = scores[:, ci]
                mask = sc > self.SCORE_THR
                if not mask.any():
                    continue
                keep = RegionProposalNetwork._nms(boxes[mask], sc[mask], self.NMS_THR)
                all_b.append(boxes[mask][keep])
                all_s.append(sc[mask][keep])
                all_l.append(
                    torch.full(
                        (len(keep),), ci, dtype=torch.long, device=props.device
                    )
                )
            if all_b:
                b = torch.cat(all_b)
                s = torch.cat(all_s)
                l = torch.cat(all_l)
                top = s.argsort(descending=True)[: self.MAX_DETS]
                results.append({"boxes": b[top], "scores": s[top], "labels": l[top]})
            else:
                results.append(
                    {
                        "boxes": torch.zeros(0, 4),
                        "scores": torch.zeros(0),
                        "labels": torch.zeros(0, dtype=torch.long),
                    }
                )
        return results

    def forward(self, images, targets=None):
        """Run the full detector; return losses in training mode or detections in eval mode."""
        img_sz = (images.shape[2], images.shape[3])
        feats = self.backbone(images)
        fpn_fs = self.fpn(feats)
        props, rpn_losses = self.rpn(fpn_fs, img_sz, targets)
        if self.training:
            s_props, s_labels, s_gt = self._sample_rois(props, targets)
            roi_feats = self.roi_align(fpn_fs[:4], s_props, img_sz)
            box_feats = self.box_head(roi_feats)
            cls_logits, bbox_preds = self.predictor(box_feats)
            cls_loss, box_loss = self._roi_loss(
                cls_logits, bbox_preds, s_labels, s_gt, s_props
            )
            return {**rpn_losses, "roi_cls": cls_loss, "roi_box": box_loss}
        else:
            roi_feats = self.roi_align(fpn_fs[:4], props, img_sz)
            box_feats = self.box_head(roi_feats)
            cls_logits, bbox_preds = self.predictor(box_feats)
            return self._postprocess(cls_logits, bbox_preds, props, img_sz), props
