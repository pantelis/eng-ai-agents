"""Shared Faster R-CNN components for the 6-notebook tutorial series.

This module centralises every reusable building block of the Faster R-CNN
detector so that the six pedagogical notebooks can focus on *explaining*
components rather than re-implementing them.  The canonical source of truth
for each implementation is **notebook 05 (training)**, which was the most
recent self-contained version at the time of extraction.

Layout
------
The module is organised into five logical sections that mirror the data flow
through the detector:

1. **Constants** — image size, class count, device, ImageNet stats, COCO names.
2. **Utilities** — IoU computation and the encode/decode box-delta transforms
   that are shared between the RPN and the ROI head.
3. **Data pipeline** — ``COCOStreamDataset`` (HuggingFace streaming) and
   ``frcnn_collate_fn``.
4. **Backbone** — ``Bottleneck``, ``ResNet50``, ``FPN`` producing multi-scale
   feature maps P2–P6.
5. **Detection head** — ``AnchorGenerator``, ``RPNHead``,
   ``RegionProposalNetwork``, ``ROIAlign``, ``TwoMLPHead``,
   ``FastRCNNPredictor``, and the end-to-end ``FasterRCNN`` wrapper.

Design decisions
----------------
* **400 × 400 input** — chosen so a full training loop (AMP, frozen backbone
  through layer 3, gradient checkpointing on layers 3–4) fits in 16 GB VRAM.
* **Pure-PyTorch NMS** — avoids a dependency on ``torchvision.ops`` so
  students can read every line.  Not speed-critical for 400 px inputs.
* **Gradient checkpointing** on ResNet layers 3 and 4 saves ~1.5 GB of
  activation memory at a modest compute cost.
* **FPN P6** is produced via max-pooling of P5 (rather than an extra backbone
  stage).  This matches the Detectron2 default for anchors > 256 px.

Notebook import mapping
-----------------------
======  ====================================================================
 NB     Imports from this module
======  ====================================================================
 01     ``COCOStreamDataset``, ``frcnn_collate_fn``, ``IMG_SIZE``,
        ``IMAGENET_MEAN``, ``IMAGENET_STD``, ``COCO_NAMES``, ``DEVICE``,
        ``NUM_CLASSES``, ``box_iou``, ``encode_boxes``
 02     ``Bottleneck``, ``ResNet50``, ``FPN``
 03     ``AnchorGenerator``, ``RPNHead``, ``RegionProposalNetwork``,
        ``decode_boxes``, ``Bottleneck``, ``ResNet50``, ``FPN``
 04     ``ROIAlign``, ``TwoMLPHead``, ``FastRCNNPredictor``,
        ``Bottleneck``, ``ResNet50``, ``FPN``, ``AnchorGenerator``,
        ``RPNHead``, ``RegionProposalNetwork``
 05     ``FasterRCNN``, ``COCOStreamDataset``, ``frcnn_collate_fn``,
        ``IMG_SIZE``, ``DEVICE``, ``IMAGENET_MEAN``, ``IMAGENET_STD``
 06     ``FasterRCNN``, ``COCO_NAMES``, ``IMAGENET_MEAN``, ``IMAGENET_STD``,
        ``IMG_SIZE``, ``DEVICE``
======  ====================================================================

``AnchorTargetGenerator`` is intentionally **not** included here — it is
defined only in notebook 01 and used nowhere else.
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

IMG_SIZE = 400
"""int: Spatial resolution (height = width) to which every COCO image is
resized before entering the network.  400 px keeps the full pipeline within
16 GB GPU memory when AMP and gradient checkpointing are enabled."""

NUM_CLASSES = 81
"""int: Total number of classes including the implicit background class at
index 0.  The 80 COCO categories occupy indices 1–80."""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""torch.device: Auto-detected compute device (CUDA when available)."""

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
"""Tensor [3,1,1]: Per-channel mean of ImageNet, used to normalise inputs so
that the frozen ResNet backbone receives data in the same distribution it was
pre-trained on."""

IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
"""Tensor [3,1,1]: Per-channel standard deviation of ImageNet."""

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
"""list[str]: Human-readable COCO category names indexed 0–80, where index 0
is the synthetic ``__background__`` class used by Faster R-CNN."""


# ─── Utilities ─────────────────────────────────────────────────────────────────


def box_iou(boxes_a, boxes_b):
    """Compute the pairwise Intersection-over-Union between two box sets.

    Both inputs are expected in ``(x1, y1, x2, y2)`` corner format.

    Parameters
    ----------
    boxes_a : Tensor, shape ``(N, 4)``
        First set of bounding boxes.
    boxes_b : Tensor, shape ``(M, 4)``
        Second set of bounding boxes.

    Returns
    -------
    Tensor, shape ``(N, M)``
        IoU matrix where element ``[i, j]`` is the IoU of ``boxes_a[i]`` with
        ``boxes_b[j]``.  A small epsilon (1e-6) is added to the denominator
        to avoid division by zero when both boxes have zero area.

    See Also
    --------
    encode_boxes : Convert matched box pairs into regression deltas.
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    ix1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    iy1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    ix2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    iy2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def encode_boxes(proposals, gt_boxes):
    """Parameterise ground-truth boxes as regression deltas relative to proposals.

    Implements the standard R-CNN box parametrisation from
    `Girshick (2015) <https://arxiv.org/abs/1504.08083>`_::

        tx = (gx - px) / pw
        ty = (gy - py) / ph
        tw = log(gw / pw)
        th = log(gh / ph)

    where ``(px, py, pw, ph)`` are the centre and size of a proposal and
    ``(gx, gy, gw, gh)`` are the centre and size of the matched GT box.

    Parameters
    ----------
    proposals : Tensor, shape ``(N, 4)``
        Reference boxes in ``(x1, y1, x2, y2)`` format.
    gt_boxes : Tensor, shape ``(N, 4)``
        Matched ground-truth boxes in the same format.

    Returns
    -------
    Tensor, shape ``(N, 4)``
        Deltas ``(tx, ty, tw, th)`` that, when applied to *proposals* via
        :func:`decode_boxes`, recover *gt_boxes*.

    See Also
    --------
    decode_boxes : Inverse operation (deltas → absolute boxes).
    """
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
    """Apply predicted regression deltas to reference boxes (inverse of :func:`encode_boxes`).

    Width/height deltas are clamped to ``±4.0`` before exponentiation to
    prevent numerical overflow when the network outputs large values during
    early training.

    Parameters
    ----------
    anchors : Tensor, shape ``(N, 4)``
        Reference boxes (anchors or proposals) in ``(x1, y1, x2, y2)`` format.
    deltas : Tensor, shape ``(N, 4)``
        Predicted deltas ``(tx, ty, tw, th)`` from the RPN or Fast R-CNN head.

    Returns
    -------
    Tensor, shape ``(N, 4)``
        Decoded boxes in ``(x1, y1, x2, y2)`` format.

    See Also
    --------
    encode_boxes : Forward operation (absolute boxes → deltas).
    """
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
    """Stream COCO from HuggingFace and resize to ``IMG_SIZE × IMG_SIZE``.

    This dataset wraps ``detection-datasets/coco`` in streaming mode so that
    the full 118k-image training set need never be downloaded to disk.  Each
    yielded sample is a ``(image_tensor, target_dict)`` pair where:

    * ``image_tensor`` is an ImageNet-normalised float32 tensor of shape
      ``(3, IMG_SIZE, IMG_SIZE)``.
    * ``target_dict`` has keys ``"boxes"`` (``float32 [K, 4]``,
      ``x1 y1 x2 y2``) and ``"labels"`` (``long [K]``, 1-indexed COCO
      categories).

    Images with no valid bounding boxes (e.g. crowd-only annotations with
    zero-area boxes after rescaling) are silently skipped.

    Parameters
    ----------
    split : str, default ``"train"``
        HuggingFace dataset split name (``"train"`` or ``"validation"``).
    max_samples : int or None, default ``None``
        If set, stop iteration after yielding this many samples.  Useful for
        quick smoke tests (e.g. ``max_samples=32``).
    """

    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        super().__init__()
        self.ds = load_dataset(
            "detection-datasets/coco", split=split, streaming=True
        )
        self.max_samples = max_samples

    def __iter__(self):
        """Yield ``(image_tensor, target_dict)`` pairs, skipping empty annotations."""
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
    """Collate function for ``DataLoader`` that stacks images but keeps targets as a list.

    Standard ``default_collate`` would fail because target dicts contain
    variable-length tensors (different images have different numbers of
    ground-truth boxes).

    Parameters
    ----------
    batch : list[tuple[Tensor, dict]]
        List of ``(image, target)`` pairs from ``COCOStreamDataset``.

    Returns
    -------
    images : Tensor, shape ``(B, 3, IMG_SIZE, IMG_SIZE)``
        Batched and stacked image tensors.
    targets : list[dict]
        Length-B list of target dictionaries, each containing ``"boxes"`` and
        ``"labels"`` tensors.
    """
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


# ─── Backbone: ResNet50 + FPN ─────────────────────────────────────────────────


class Bottleneck(nn.Module):
    """ResNet bottleneck residual block (1 × 1 → 3 × 3 → 1 × 1 convolutions).

    This is the standard "bottleneck" building block used in ResNet-50 and
    deeper variants.  The block narrows to ``out_ch`` channels via the first
    1 × 1 conv, applies a spatial 3 × 3 conv, and expands back to
    ``out_ch * 4`` via the final 1 × 1 conv.  A skip connection adds the
    input directly to the output (with an optional linear projection when
    spatial resolution or channel count changes).

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Bottleneck width (the narrow channel count).  The output channel
        count is ``out_ch * expansion`` (= ``out_ch * 4``).
    stride : int, default 1
        Spatial stride applied in the 3 × 3 convolution.  Set to 2 for the
        first block in stages 2–4 to halve the feature map resolution.
    downsample : nn.Module or None, default ``None``
        Optional 1 × 1 conv + BN projection applied to the shortcut path
        when the input and output shapes differ.

    Attributes
    ----------
    expansion : int
        Class-level constant = 4, giving the ratio of output to bottleneck
        channels.
    """

    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * 4)
        self.downsample = downsample

    def forward(self, x):
        """Run the bottleneck convolutions and fuse with the residual shortcut.

        Parameters
        ----------
        x : Tensor, shape ``(B, in_ch, H, W)``

        Returns
        -------
        Tensor, shape ``(B, out_ch * 4, H', W')``
            Where ``H' = H // stride`` and ``W' = W // stride``.
        """
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(x)
        return F.relu(out + identity)


class ResNet50(nn.Module):
    """Minimal ResNet-50 backbone returning multi-scale features C2–C5.

    The architecture follows `He et al. (2016) <https://arxiv.org/abs/1512.03385>`_
    with four residual stages (layer1–layer4) preceded by a 7 × 7-conv stem
    and max-pool.  Gradient checkpointing is applied to layers 3 and 4 via
    ``torch.utils.checkpoint`` to reduce activation memory at the cost of one
    extra forward pass per checkpointed segment.

    Output feature maps and their spatial strides relative to the input:

    ======  ========  ===========
    Output  Channels  Stride
    ======  ========  ===========
    C2      256       4×
    C3      512       8×
    C4      1024      16×
    C5      2048      32×
    ======  ========  ===========
    """

    def __init__(self):
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
        """Create one ResNet stage with the given number of bottleneck blocks.

        Parameters
        ----------
        in_ch : int
            Input channel count (from the previous stage's output).
        out_ch : int
            Bottleneck width for this stage.
        blocks : int
            Number of bottleneck blocks in the stage.
        stride : int
            Spatial stride of the first block (1 for layer1, 2 otherwise).

        Returns
        -------
        nn.Sequential
            The composed stage.
        """
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
        """Extract C2–C5 feature maps; layers 3–4 use gradient checkpointing.

        Parameters
        ----------
        x : Tensor, shape ``(B, 3, H, W)``
            ImageNet-normalised input image batch.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            ``(C2, C3, C4, C5)`` feature maps at strides 4, 8, 16, 32.
        """
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = grad_ckpt(self.layer3, c3, use_reentrant=False)
        c5 = grad_ckpt(self.layer4, c4, use_reentrant=False)
        return c2, c3, c4, c5


class FPN(nn.Module):
    """Feature Pyramid Network producing semantically rich multi-scale features.

    Implements the FPN from `Lin et al. (2017) <https://arxiv.org/abs/1612.03144>`_
    with a top-down pathway and lateral connections.  The backbone feature maps
    C2–C5 are fused into pyramid levels P2–P5, and a P6 level is added via
    max-pooling of P5 for large-anchor detection.

    Parameters
    ----------
    in_channels : tuple[int, ...], default ``(256, 512, 1024, 2048)``
        Channel counts of the backbone outputs C2–C5.
    out_channels : int, default 256
        Uniform channel count for all pyramid levels.

    Notes
    -----
    ``forward()`` expects a **tuple** of four feature maps, not four
    positional arguments.  Call as ``fpn((c2, c3, c4, c5))``, not
    ``fpn(c2, c3, c4, c5)``.
    """

    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256):
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels]
        )
        self.output = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels]
        )
        self.p6 = nn.MaxPool2d(1, stride=2)

    def forward(self, features):
        """Fuse backbone features via top-down pathway and return P2–P6.

        Parameters
        ----------
        features : tuple[Tensor, Tensor, Tensor, Tensor]
            ``(C2, C3, C4, C5)`` from the backbone.

        Returns
        -------
        list[Tensor]
            ``[P2, P3, P4, P5, P6]`` — five pyramid levels at strides
            4, 8, 16, 32, 64 respectively.
        """
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
    """Generate multi-scale, multi-aspect-ratio anchors tiled over FPN grids.

    For each FPN level, a set of base anchors is created at every grid cell
    centre.  The base anchors are defined by one ``anchor_size`` and three
    ``aspect_ratios``, giving ``k = len(aspect_ratios)`` anchors per cell.

    Parameters
    ----------
    anchor_sizes : tuple[int, ...], default ``(32, 64, 128, 256, 512)``
        Anchor side lengths (in pixels at the *input image* scale), one per
        FPN level P2–P6.
    aspect_ratios : tuple[float, ...], default ``(0.5, 1.0, 2.0)``
        Width / height ratios applied to every anchor size.
    strides : tuple[int, ...], default ``(4, 8, 16, 32, 64)``
        Feature-map stride (in input-image pixels) for each FPN level.
    """

    def __init__(
        self,
        anchor_sizes=(32, 64, 128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        strides=(4, 8, 16, 32, 64),
    ):
        super().__init__()
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides

    def _base(self, size):
        """Create ``k`` canonical anchors centred at the origin for one scale.

        Parameters
        ----------
        size : int
            Anchor side length in pixels.

        Returns
        -------
        Tensor, shape ``(k, 4)``
            Base anchors in ``(x1, y1, x2, y2)`` format, centred at (0, 0).
        """
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
        """Tile base anchors across every FPN grid cell and concatenate.

        Parameters
        ----------
        feature_maps : list[Tensor]
            FPN outputs ``[P2, P3, P4, P5, P6]``.
        image_size : tuple[int, int]
            ``(H, W)`` of the input image (used only for device placement).

        Returns
        -------
        Tensor, shape ``(A, 4)``
            Concatenated anchors across all levels, where
            ``A = sum(H_l * W_l * k)`` for each level ``l``.
        """
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
    """Shared convolutional head that predicts objectness and box deltas per anchor.

    A single 3 × 3 conv (shared across FPN levels) followed by two sibling
    1 × 1 convs: one for binary objectness logits and one for 4-d box
    regression deltas.  Weights are initialised with ``std=0.01`` (normal)
    and zero biases, following the Faster R-CNN paper convention.

    Parameters
    ----------
    in_ch : int, default 256
        Number of input channels (= FPN output channels).
    k : int, default 3
        Number of anchors per spatial location (= number of aspect ratios).
    """

    def __init__(self, in_ch=256, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.cls = nn.Conv2d(in_ch, k, 1)
        self.box = nn.Conv2d(in_ch, k * 4, 1)
        for layer in [self.conv, self.cls, self.box]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        """Apply the RPN head independently to each FPN level.

        Parameters
        ----------
        features : list[Tensor]
            FPN outputs, each of shape ``(B, 256, H_l, W_l)``.

        Returns
        -------
        cls_outs : list[Tensor]
            Per-level objectness logits, each ``(B, k, H_l, W_l)``.
        box_outs : list[Tensor]
            Per-level box deltas, each ``(B, k*4, H_l, W_l)``.
        """
        cls_outs, box_outs = [], []
        for f in features:
            t = F.relu(self.conv(f))
            cls_outs.append(self.cls(t))
            box_outs.append(self.box(t))
        return cls_outs, box_outs


class RegionProposalNetwork(nn.Module):
    """Complete RPN: anchor generation, scoring, filtering, NMS, and training losses.

    During **training**, the RPN:

    1. Generates anchors over the FPN grid.
    2. Predicts objectness scores and box deltas via :class:`RPNHead`.
    3. Decodes proposals and filters them (clip, size threshold, top-k, NMS).
    4. Assigns anchors to GT boxes (IoU ≥ 0.7 → positive, < 0.3 → negative)
       and subsamples a balanced mini-batch of 256 anchors (50 % positive).
    5. Returns proposals **and** RPN losses (``rpn_cls``, ``rpn_box``).

    During **inference**, only steps 1–3 are executed and losses are empty.

    Parameters
    ----------
    head : RPNHead
        Convolutional prediction head.
    anchor_gen : AnchorGenerator
        Anchor-tiling module.
    pre_nms : int, default 2000
        Keep only the top-``pre_nms`` proposals (by objectness score) before NMS.
    post_nms : int, default 1000
        Keep at most ``post_nms`` proposals after NMS.
    nms_thr : float, default 0.7
        IoU threshold for non-maximum suppression of proposals.
    min_sz : int, default 16
        Minimum side length (in pixels) for a proposal to be kept.

    Attributes
    ----------
    RPN_BATCH : int
        Mini-batch size for RPN anchor sampling (256).
    POS_FRAC : float
        Target fraction of positive anchors in the mini-batch (0.5).
    POS_THR : float
        IoU threshold above which an anchor is labelled positive (0.7).
    NEG_THR : float
        IoU threshold below which an anchor is labelled negative (0.3).
    """

    RPN_BATCH = 256
    POS_FRAC = 0.5
    POS_THR = 0.7
    NEG_THR = 0.3

    def __init__(
        self, head, anchor_gen, pre_nms=2000, post_nms=1000, nms_thr=0.7, min_sz=16
    ):
        super().__init__()
        self.head = head
        self.anchor_gen = anchor_gen
        self.pre_nms = pre_nms
        self.post_nms = post_nms
        self.nms_thr = nms_thr
        self.min_sz = min_sz

    def _filter(self, props, scores, img_size):
        """Clip proposals to the image, discard tiny boxes, rank by score, and apply NMS.

        Parameters
        ----------
        props : Tensor, shape ``(N, 4)``
            Decoded proposal boxes.
        scores : Tensor, shape ``(N,)``
            Objectness scores (after sigmoid).
        img_size : tuple[int, int]
            ``(H, W)`` of the input image for clipping.

        Returns
        -------
        props : Tensor, shape ``(K, 4)``
            Filtered proposals (K ≤ ``post_nms``).
        scores : Tensor, shape ``(K,)``
            Corresponding scores.
        """
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
        """Pure-PyTorch greedy non-maximum suppression.

        Iteratively selects the highest-scoring box, suppresses all boxes
        with IoU > ``thr`` against it, and repeats.

        Parameters
        ----------
        boxes : Tensor, shape ``(N, 4)``
            Bounding boxes in ``(x1, y1, x2, y2)`` format.
        scores : Tensor, shape ``(N,)``
            Confidence scores corresponding to each box.
        thr : float
            IoU suppression threshold.

        Returns
        -------
        Tensor, shape ``(K,)``
            Indices of kept boxes, in descending score order.
        """
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
        """Match anchors to GT boxes and subsample a balanced mini-batch.

        Assignment rules (following the original Faster R-CNN paper):

        * IoU ≥ 0.7 with *any* GT → positive (label = 1).
        * IoU < 0.3 with *all* GTs → negative (label = 0).
        * 0.3 ≤ IoU < 0.7 → ignored (label = −1, excluded from loss).
        * The anchor with highest IoU to each GT is forced positive
          (ensures every GT has at least one matching anchor).

        The mini-batch is then subsampled to at most ``RPN_BATCH`` anchors
        with at most ``POS_FRAC`` positives; excess positives or negatives
        are set to label −1 (ignored).

        Parameters
        ----------
        anchors : Tensor, shape ``(A, 4)``
            All anchors across all FPN levels.
        gt_boxes : Tensor, shape ``(G, 4)``
            Ground-truth boxes for one image.

        Returns
        -------
        labels : Tensor, shape ``(A,)``
            1 = positive, 0 = negative, −1 = ignored.
        matched_gt : Tensor, shape ``(A, 4)``
            The GT box matched to each anchor (only meaningful where
            ``labels == 1``).
        """
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
        """Run the RPN: predict, decode, filter, and (optionally) compute losses.

        Parameters
        ----------
        features : list[Tensor]
            FPN feature maps ``[P2, P3, P4, P5, P6]``.
        image_size : tuple[int, int]
            ``(H, W)`` of the input images.
        targets : list[dict] or None
            Ground-truth targets (only needed during training).  Each dict
            must contain ``"boxes"`` (Tensor ``[G, 4]``).

        Returns
        -------
        proposals : list[Tensor]
            Length-B list of proposal tensors, each ``(K_i, 4)``.
        losses : dict[str, Tensor]
            Empty during inference.  During training, contains:
            ``"rpn_cls"`` (binary cross-entropy on sampled anchors) and
            ``"rpn_box"`` (smooth-L1 on positive anchors).
        """
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
    """ROI Align via bilinear ``grid_sample``, with FPN level assignment.

    Each proposal is assigned to an FPN level using the formula from
    `Lin et al. (2017) <https://arxiv.org/abs/1612.03144>`_::

        k = floor(k0 + log2(sqrt(area) / 224))

    clamped to ``[k_min, k_max]``.  The proposal's bounding box is then
    mapped to a normalised grid on the assigned feature map, and
    ``F.grid_sample`` performs bilinear interpolation to produce a fixed-size
    ``(out_size × out_size)`` feature crop.

    Parameters
    ----------
    out_size : int, default 7
        Spatial resolution of the output feature map (7 × 7 for standard
        Fast R-CNN).
    k0 : int, default 4
        Canonical FPN level for a 224 × 224 ROI.
    k_min : int, default 2
        Minimum FPN level index (P2).
    k_max : int, default 5
        Maximum FPN level index (P5).  Proposals larger than the P5 receptive
        field are still pooled from P5 (which is the lowest-resolution
        feature map used for ROI pooling — P6 is only used by the RPN).
    """

    def __init__(self, out_size=7, k0=4, k_min=2, k_max=5):
        super().__init__()
        self.out_size = out_size
        self.k0 = k0
        self.k_min = k_min
        self.k_max = k_max

    def _level(self, boxes):
        """Map each proposal to its target FPN level index (0-based, relative to k_min).

        Parameters
        ----------
        boxes : Tensor, shape ``(N, 4)``
            Proposal boxes in ``(x1, y1, x2, y2)`` format.

        Returns
        -------
        Tensor, shape ``(N,)``
            Integer level indices in ``[0, k_max - k_min]``.
        """
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
        """Pool each proposal from its assigned FPN level into a fixed-size feature map.

        Parameters
        ----------
        fmaps : list[Tensor]
            FPN feature maps ``[P2, P3, P4, P5]`` (P6 is not used for ROI
            pooling).
        proposals : list[Tensor]
            Length-B list of proposal tensors, each ``(K_i, 4)``.
        image_size : tuple[int, int]
            ``(H, W)`` of the input image, used to normalise box coordinates
            to the ``[-1, 1]`` range expected by ``grid_sample``.

        Returns
        -------
        Tensor, shape ``(sum(K_i), C, out_size, out_size)``
            Pooled features for all proposals across the batch.
        """
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
    """Two-layer MLP that projects flattened ROI features to a shared representation.

    This is the "box head" from Fast R-CNN: the pooled 7 × 7 feature maps are
    flattened to ``256 * 7 * 7 = 12544`` dimensions and passed through two
    fully connected layers with ReLU activations.

    Parameters
    ----------
    in_channels : int, default ``256 * 7 * 7``
        Flattened input dimensionality (= FPN channels × pool height × pool
        width).
    fc_dim : int, default 1024
        Hidden and output dimensionality of the two FC layers.
    """

    def __init__(self, in_channels=256 * 7 * 7, fc_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)

    def forward(self, x):
        """Flatten and project ROI features.

        Parameters
        ----------
        x : Tensor, shape ``(N, C, pool_h, pool_w)``
            Pooled ROI features from :class:`ROIAlign`.

        Returns
        -------
        Tensor, shape ``(N, fc_dim)``
            Shared representation for classification and regression.
        """
        return F.relu(self.fc2(F.relu(self.fc1(x.flatten(1)))))


class FastRCNNPredictor(nn.Module):
    """Final prediction layer: class logits and class-specific box deltas.

    Two parallel linear layers map the shared ``TwoMLPHead`` representation to:

    * **Class logits** — ``(N, num_classes)`` used with cross-entropy loss.
    * **Box deltas** — ``(N, num_classes * 4)`` giving class-specific
      ``(tx, ty, tw, th)`` refinements for each proposal.

    Weights are initialised with small normal noise (``std=0.01`` for cls,
    ``std=0.001`` for box) and zero biases, following Detectron2 conventions.

    Parameters
    ----------
    in_channels : int, default 1024
        Input dimensionality (= ``fc_dim`` of :class:`TwoMLPHead`).
    num_classes : int, default 81
        Number of output classes including background.
    """

    def __init__(self, in_channels=1024, num_classes=81):
        super().__init__()
        self.cls = nn.Linear(in_channels, num_classes)
        self.box = nn.Linear(in_channels, num_classes * 4)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.zeros_(self.cls.bias)
        nn.init.normal_(self.box.weight, std=0.001)
        nn.init.zeros_(self.box.bias)

    def forward(self, x):
        """Predict class logits and box deltas for each ROI.

        Parameters
        ----------
        x : Tensor, shape ``(N, in_channels)``
            Shared feature representation from :class:`TwoMLPHead`.

        Returns
        -------
        cls_logits : Tensor, shape ``(N, num_classes)``
            Raw (pre-softmax) class scores.
        bbox_preds : Tensor, shape ``(N, num_classes * 4)``
            Class-specific box regression deltas.
        """
        return self.cls(x), self.box(x)


# ─── Full model ────────────────────────────────────────────────────────────────


class FasterRCNN(nn.Module):
    """End-to-end Faster R-CNN detector combining all sub-modules.

    Assembles the full detection pipeline:

    1. **Backbone** (``ResNet50``) → C2–C5 multi-scale features.
    2. **FPN** → P2–P6 pyramid levels.
    3. **RPN** → object proposals + RPN losses (training only).
    4. **ROI Align** → fixed-size features per proposal.
    5. **Box head** (``TwoMLPHead`` + ``FastRCNNPredictor``) → class logits
       and box deltas + Fast R-CNN losses (training only).
    6. **Post-processing** → per-class NMS, score thresholding, top-100
       detections (inference only).

    Frozen layers
    ~~~~~~~~~~~~~
    The ResNet stem and layers 1–3 are frozen (``requires_grad_(False)``).
    Only layer 4, the FPN, RPN, and ROI heads are trained.  This saves ~60 %
    of the activation memory compared to training the full backbone.

    Training vs. inference
    ~~~~~~~~~~~~~~~~~~~~~~
    * **Training** (``model.train()``): ``forward()`` returns a loss dict
      with keys ``"rpn_cls"``, ``"rpn_box"``, ``"roi_cls"``, ``"roi_box"``.
    * **Inference** (``model.eval()``): ``forward()`` returns a tuple
      ``(results, proposals)`` where ``results`` is a list of detection dicts
      (one per image) and ``proposals`` is the raw RPN output (useful for
      visualising proposal quality in notebook 06).

    Parameters
    ----------
    num_classes : int, default 81
        Number of detection classes including background.

    Attributes
    ----------
    ROI_BATCH : int
        Number of ROIs sampled per image for the Fast R-CNN head (512).
    ROI_POS_FRAC : float
        Target positive fraction in the ROI mini-batch (0.25).
    ROI_POS_THR : float
        IoU threshold for a proposal to be labelled positive (0.5).
    SCORE_THR : float
        Minimum score for a detection to survive post-processing (0.05).
    NMS_THR : float
        IoU threshold for per-class NMS in post-processing (0.5).
    MAX_DETS : int
        Maximum number of detections kept per image (100).
    """

    ROI_BATCH = 512
    ROI_POS_FRAC = 0.25
    ROI_POS_THR = 0.5
    SCORE_THR = 0.05
    NMS_THR = 0.5
    MAX_DETS = 100

    def __init__(self, num_classes=81):
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
        """Sample a balanced mini-batch of ROIs for the Fast R-CNN head.

        For each image, proposals are concatenated with GT boxes (to guarantee
        some positive examples), matched to GT via IoU, and subsampled to
        ``ROI_BATCH`` ROIs with at most ``ROI_POS_FRAC`` positives.

        Parameters
        ----------
        proposals : list[Tensor]
            RPN proposals, one tensor ``(K_i, 4)`` per image.
        targets : list[dict]
            Ground-truth targets with ``"boxes"`` and ``"labels"``.

        Returns
        -------
        s_props : list[Tensor]
            Sampled proposals per image.
        s_labels : list[Tensor]
            Class labels for sampled proposals (0 = background).
        s_gt : list[Tensor]
            Matched GT boxes for sampled proposals.
        """
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
        """Compute Fast R-CNN classification and box-regression losses.

        * **Classification**: cross-entropy over all sampled ROIs (positive
          and negative).
        * **Box regression**: smooth-L1 (with ``beta = 1/9``) on positive
          ROIs only, using class-specific deltas.

        Parameters
        ----------
        cls_logits : Tensor, shape ``(R, num_classes)``
            Predicted class logits for all sampled ROIs.
        bbox_preds : Tensor, shape ``(R, num_classes * 4)``
            Predicted class-specific box deltas.
        labels_list : list[Tensor]
            True class labels per image.
        gt_list : list[Tensor]
            Matched GT boxes per image.
        props_list : list[Tensor]
            Sampled proposals per image.

        Returns
        -------
        cls_loss : Tensor
            Scalar cross-entropy loss.
        box_loss : Tensor
            Scalar smooth-L1 regression loss (zero if no positives).
        """
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
        """Decode predictions and apply per-class NMS to produce final detections.

        For each class (excluding background), box deltas are decoded relative
        to the proposals, clipped to the image boundary, score-thresholded,
        and NMS-filtered.  The top ``MAX_DETS`` detections (by score) are
        returned.

        Parameters
        ----------
        cls_logits : Tensor, shape ``(R, num_classes)``
            Class logits for all proposals.
        bbox_preds : Tensor, shape ``(R, num_classes * 4)``
            Class-specific box deltas.
        proposals_list : list[Tensor]
            Proposals per image (from the RPN).
        image_size : tuple[int, int]
            ``(H, W)`` for box clipping.

        Returns
        -------
        list[dict]
            Length-B list of detection dictionaries, each containing:

            * ``"boxes"`` — Tensor ``(D, 4)`` in ``(x1, y1, x2, y2)`` format.
            * ``"scores"`` — Tensor ``(D,)`` confidence scores.
            * ``"labels"`` — Tensor ``(D,)`` predicted class indices (1-indexed).
        """
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
        """Run the full Faster R-CNN pipeline.

        Parameters
        ----------
        images : Tensor, shape ``(B, 3, H, W)``
            Batch of ImageNet-normalised images.
        targets : list[dict] or None
            Ground-truth annotations (required during training).  Each dict
            must have ``"boxes"`` (Tensor ``[G, 4]``) and ``"labels"``
            (Tensor ``[G]``).

        Returns
        -------
        dict[str, Tensor]  *(training mode)*
            Loss dictionary with keys ``"rpn_cls"``, ``"rpn_box"``,
            ``"roi_cls"``, ``"roi_box"``.
        tuple[list[dict], list[Tensor]]  *(eval mode)*
            ``(results, proposals)`` where ``results`` is a length-B list of
            detection dicts and ``proposals`` is the raw RPN output (for
            visualisation in notebook 06).
        """
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
