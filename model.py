"""
Network definition shared by ``train.py`` and ``predict.py``.

Late-fusion network for canopy height model (CHM) regression, one branch per
sensor, each branch ingesting its patch at the sensor's NATIVE resolution::

    Sentinel-2  (10 bands @ 10 m)  -> U-Net, ResNet-50 encoder (ImageNet init) \
    ALOS PALSAR (HH, HV  @ 25 m)   -> light U-Net (SensorBackbone)              \  gated
    Sentinel-1  (VV, VH  @ 10 m)   -> light U-Net (SensorBackbone)              /  fusion -> head -> CHM
    DEM         (1 band  @ 30 m)   -> light U-Net (SensorBackbone)             /

Feature maps of the auxiliary branches are resampled (bilinear) to the
Sentinel-2 grid *inside* the network, so no input image is ever resampled.
The network returns a logit map; the caller applies a sigmoid and multiplies by
``chm_max`` to obtain metres.

Module/attribute names (``backbones``, ``fusion``, ``reg_head``, ``inc``,
``downs`` ...) are kept identical to the original research code so that
checkpoints trained with it load unchanged (see ``config_from_checkpoint``).
"""
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

# Keys of a sensor entry inside ``config["sensors"]``.
SENSOR_CONFIG_KEYS = ("name", "channels", "resolution_m", "patch_side",
                      "base_channels", "depth", "min_per_band", "max_per_band")

# Names used by the original (Portuguese) code -> names used here.
LEGACY_SENSOR_NAMES = {"MDT": "DEM"}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(conv3x3 -> BN -> ReLU) x 2."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


def auto_backbone_depth(side, cap=4, min_bottleneck=4):
    """Number of 2x down-sampling stages a light backbone can have for a patch
    of ``side`` pixels without degenerating the bottleneck.

    Each stage halves the side; we stop when the side would become odd or
    smaller than ``min_bottleneck``::

        256 px -> 4   128 px -> 4   102 px -> 1 (51 is odd)   85 px -> 1
    """
    depth, s = 0, int(side)
    while s % 2 == 0 and s // 2 >= min_bottleneck and depth < cap:
        s //= 2
        depth += 1
    return max(1, depth)


class SensorBackbone(nn.Module):
    """Light U-Net used by the auxiliary branches (SAR, DEM).

    Takes a patch of ``in_channels`` bands at the sensor's native resolution
    and returns ``out_channels`` feature maps on the SAME grid. ``depth``
    should come from ``auto_backbone_depth(patch_side)``.
    """

    def __init__(self, in_channels, out_channels=64, base_channels=32, depth=3):
        super().__init__()
        assert 1 <= depth <= 5, f"depth out of range: {depth}"
        self.depth = depth
        bc = base_channels
        self.inc = DoubleConv(in_channels, bc)
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(bc * (2 ** i), bc * (2 ** (i + 1))),
            ))
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(depth, 0, -1):
            ic, oc = bc * (2 ** i), bc * (2 ** (i - 1))
            self.ups.append(nn.ConvTranspose2d(ic, oc, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(ic, oc))
        self.outc = nn.Conv2d(bc, out_channels, kernel_size=1)

    def forward(self, x):
        feats = [self.inc(x)]
        for down in self.downs:
            feats.append(down(feats[-1]))
        x = feats[-1]
        for j, (up, conv) in enumerate(zip(self.ups, self.up_convs)):
            x = up(x)
            skip = feats[-(j + 2)]
            # odd sizes: pad the up-sampled map to the skip connection size
            dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
            x = conv(torch.cat([skip, x], dim=1))
        return self.outc(x)


class GatedFusion(nn.Module):
    """Gated cross-modal fusion for N modalities.

    A global descriptor (GAP) of the concatenated features conditions a small
    MLP that outputs one gate per channel and per modality. Each modality is
    re-weighted by its gate, concatenated and refined by a 3x3 convolution.
    """

    def __init__(self, feat_channels, n_modalities, reduction=8):
        super().__init__()
        self.feat = feat_channels
        self.n = n_modalities
        total = n_modalities * feat_channels
        hidden = max(1, total // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(total, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, total, 1),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(total, total, 3, padding=1),
            nn.BatchNorm2d(total),
            nn.ReLU(inplace=True),
        )
        self.out_channels = total
        self.last_gate = None  # (B, N*feat, 1, 1) of the last forward, for diagnostics

    def forward(self, feats):
        cat = torch.cat(feats, dim=1)
        gate = torch.sigmoid(self.mlp(self.pool(cat)))
        self.last_gate = gate.detach()
        gated = [f * gate[:, i * self.feat:(i + 1) * self.feat] for i, f in enumerate(feats)]
        return self.mix(torch.cat(gated, dim=1))

    def gate_stats(self):
        """Mean gate (0-1) per modality in the last forward pass."""
        if self.last_gate is None:
            return None
        return [self.last_gate[:, i * self.feat:(i + 1) * self.feat].mean().item()
                for i in range(self.n)]


def build_s2_backbone(in_channels, out_channels, encoder_name="resnet50", pretrained=True):
    """Sentinel-2 branch: U-Net with a ResNet-50 encoder (4 stages, 1/16),
    wide decoder and scSE attention, returning ``out_channels`` feature maps."""
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=out_channels,
        activation=None,
        encoder_depth=4,
        decoder_channels=(512, 256, 128, 64),
        decoder_attention_type="scse",
    )


class MultiSensorCHMNet(nn.Module):
    """Late-fusion regression network.

    ``forward(x)`` takes a list of tensors, one per branch in the order of
    ``modality_names`` (Sentinel-2 first), and returns a logit map
    ``(B, 1, H, W)`` on the Sentinel-2 grid.
    """

    def __init__(self, backbones, modality_names, feat_channels=64,
                 head_channels=64, align_mode="bilinear"):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        self.modality_names = list(modality_names)
        self.n_modalities = len(self.backbones)
        self.align_mode = align_mode
        self.fusion = GatedFusion(feat_channels, self.n_modalities)
        self.reg_head = nn.Sequential(
            nn.Conv2d(self.fusion.out_channels, head_channels, 3, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def gate_stats(self):
        return self.fusion.gate_stats()

    def _align(self, f, size):
        if f.shape[2:] == size:
            return f
        kwargs = {"align_corners": False} if self.align_mode in ("bilinear", "bicubic") else {}
        return F.interpolate(f, size=size, mode=self.align_mode, **kwargs)

    def forward(self, x):
        if torch.is_tensor(x):
            x = [x]
        if len(x) != self.n_modalities:
            raise ValueError(f"expected {self.n_modalities} inputs "
                             f"({', '.join(self.modality_names)}), got {len(x)}")
        feats = [bb(xi) for bb, xi in zip(self.backbones, x)]
        ref_hw = feats[0].shape[2:]
        feats = [feats[0]] + [self._align(f, ref_hw) for f in feats[1:]]
        return self.reg_head(self.fusion(feats))


# ---------------------------------------------------------------------------
# Config <-> model
# ---------------------------------------------------------------------------
def build_model(config, pretrained=False):
    """Instantiate the network described by ``config`` (see train.py for the
    layout). ``pretrained`` only matters when training from scratch."""
    feat = int(config["feature_channels"])
    backbones = [build_s2_backbone(int(config["s2"]["channels"]), feat,
                                   config.get("encoder_name", "resnet50"), pretrained)]
    for s in config["sensors"]:
        backbones.append(SensorBackbone(int(s["channels"]), feat,
                                        int(s["base_channels"]), int(s["depth"])))
    names = ["S2"] + [s["name"] for s in config["sensors"]]
    return MultiSensorCHMNet(backbones, names, feat, int(config["head_channels"]),
                             config.get("align_mode", "bilinear"))


def config_from_checkpoint(ck):
    """Return the model/normalisation config stored in a checkpoint.

    Checkpoints written by ``train.py`` carry it under ``"config"``. Checkpoints
    written by the original research code are converted from their legacy keys
    (``sar_norm``, ``modality_order``, ``rgb_divisor`` ...).
    """
    if "config" in ck:
        return ck["config"]
    if "sar_norm" not in ck:
        raise KeyError("checkpoint has neither 'config' nor the legacy 'sar_norm' entry")
    if not ck.get("fusion_mode", True) or ck.get("fusion", "gated") != "gated":
        raise ValueError("only gated-fusion checkpoints are supported by this repository")

    sensors = []
    for name in [m for m in ck["modality_order"] if m != "S2"]:
        meta = ck["sar_norm"][name]
        side = int(meta["patch_side"])
        sensors.append(dict(
            name=LEGACY_SENSOR_NAMES.get(name, name),
            channels=int(meta["channels"]),
            resolution_m=float(meta["resolution_m"]),
            patch_side=side,
            base_channels=32,
            depth=auto_backbone_depth(side),
            min_per_band=[float(v) for v in meta["min_per_band"]],
            max_per_band=[float(v) for v in meta["max_per_band"]],
        ))
    chm_max = ck["max_chm_value"]
    chm_max = float(chm_max[0] if isinstance(chm_max, (list, tuple)) else chm_max)
    return dict(
        # the legacy checkpoint stores the S2 encoder name only indirectly
        # (the S1 branch inherits it); every published model used resnet50.
        encoder_name=ck.get("s1_encoder_name") or "resnet50",
        feature_channels=int(ck.get("feature_channels", 64)),
        head_channels=int(ck.get("head_channels", 64)),
        align_mode=ck.get("feat_align_mode", "bilinear"),
        patch_size=int(ck.get("psize", 256)),
        s2=dict(channels=int(ck["rgb_channels"]),
                resolution_m=float(ck.get("s2_resolution", 10.0)),
                divisor=float(ck["rgb_divisor"])),
        chm_max=chm_max,
        sensors=sensors,
    )


def load_model(checkpoint_path, device):
    """Rebuild the network from a checkpoint and load its weights.

    Returns ``(model, config, checkpoint_dict)`` with the model in eval mode.
    """
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = config_from_checkpoint(ck)
    model = build_model(config, pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), config, ck


# ---------------------------------------------------------------------------
# Normalisation (identical in training and inference)
# ---------------------------------------------------------------------------
def normalize_s2(arr, divisor):
    """Sentinel-2 reflectance (C, H, W) / divisor, clipped to [0, 1]."""
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 2:
        arr = arr[None]
    return np.clip(arr / float(divisor), 0.0, 1.0)


def normalize_sensor(arr, min_per_band, max_per_band):
    """Per-band min/max scaling of an auxiliary patch (C, h, w) to [0, 1]."""
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 2:
        arr = arr[None]
    c = arr.shape[0]
    mn = np.asarray(min_per_band, dtype=np.float32)[:c].reshape(c, 1, 1)
    mx = np.asarray(max_per_band, dtype=np.float32)[:c].reshape(c, 1, 1)
    return np.clip((arr - mn) / np.maximum(mx - mn, 1e-6), 0.0, 1.0)
