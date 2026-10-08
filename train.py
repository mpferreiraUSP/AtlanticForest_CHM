#!/usr/bin/env python3
"""
Train the multi-sensor canopy height network.

Point the script at the patch folders (one folder per modality, ``.npy`` files
with the same stem in every folder) and it takes care of the rest: pairing,
train/validation split, normalisation statistics, training with early
stopping, and writing ``best.pth`` / ``last.pth`` + ``config.json`` to
``--output_dir``. The auxiliary branches (Sentinel-1, ALOS PALSAR, DEM) are
enabled simply by passing their folder.

Examples
--------
Sentinel-2 + Sentinel-1 + ALOS PALSAR + DEM::

    python train.py --s2_dir data/train/S2 --chm_dir data/train/CHM \
        --s1_dir data/train/S1 --alos_dir data/train/ALOS --dem_dir data/train/DEM \
        --output_dir runs/s2_s1_alos_dem

Sentinel-2 + Sentinel-1 only::

    python train.py --s2_dir data/train/S2 --chm_dir data/train/CHM \
        --s1_dir data/train/S1 --output_dir runs/s2_s1

Fine-tune from a previous run::

    python train.py ... --init_from_checkpoint runs/s2_s1_alos_dem/best.pth --lr 1e-4
"""
import argparse
import glob
import json
import logging
import math
import os
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend; figures are only written to disk
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import (SENSOR_CONFIG_KEYS, auto_backbone_depth, build_model,
                   normalize_s2, normalize_sensor)

# Auxiliary branches: (name, CLI folder argument, CLI resolution argument).
# The order defines the order of the branches inside the network.
SENSORS = [
    ("ALOS", "alos_dir", "alos_resolution"),
    ("S1", "s1_dir", "s1_resolution"),
    ("DEM", "dem_dir", "dem_resolution"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Train the multi-sensor CHM network.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # -- data -----------------------------------------------------------------
    p.add_argument("--s2_dir", required=True, help="Sentinel-2 patches (.npy, (C, H, W)).")
    p.add_argument("--chm_dir", required=True, help="Reference CHM patches (.npy, (H, W), metres).")
    p.add_argument("--s1_dir", default=None, help="Sentinel-1 patches; enables the S1 branch.")
    p.add_argument("--alos_dir", default=None, help="ALOS PALSAR patches; enables the ALOS branch.")
    p.add_argument("--dem_dir", default=None, help="DEM patches; enables the DEM branch.")
    p.add_argument("--output_dir", required=True, help="Where checkpoints, logs and figures go.")
    p.add_argument("--s2_divisor", type=float, default=10000.0,
                   help="Sentinel-2 bands are divided by this value (reflectance x 10000).")
    p.add_argument("--chm_max", type=float, default=100.0,
                   help="CHM is divided by this value (metres); the network predicts [0, 1].")
    p.add_argument("--dem_divisor", type=float, default=2100.0,
                   help="DEM elevation is divided by this value (metres). S1/ALOS use per-band "
                        "min/max computed over the training patches instead.")
    p.add_argument("--s2_resolution", type=float, default=10.0, help="m/pixel (reference grid).")
    p.add_argument("--s1_resolution", type=float, default=10.0, help="m/pixel.")
    p.add_argument("--alos_resolution", type=float, default=25.0, help="m/pixel.")
    p.add_argument("--dem_resolution", type=float, default=30.0, help="m/pixel.")
    p.add_argument("--val_fraction", type=float, default=0.2, help="Fraction of patches held out for validation.")
    p.add_argument("--augment", action="store_true", help="Random flips / 90-degree rotations.")
    p.add_argument("--workers", type=int, default=8, help="DataLoader workers.")
    p.add_argument("--seed", type=int, default=31)
    # -- model ----------------------------------------------------------------
    p.add_argument("--encoder_name", default="resnet50", help="Encoder of the Sentinel-2 U-Net (any smp encoder).")
    p.add_argument("--no_pretrained", action="store_true", help="Do not initialise the S2 encoder with ImageNet weights.")
    p.add_argument("--feature_channels", type=int, default=64, help="Feature maps produced by every branch.")
    p.add_argument("--head_channels", type=int, default=64, help="Width of the regression head.")
    p.add_argument("--sensor_base_channels", type=int, default=32, help="Base width of the auxiliary branches.")
    # -- optimisation ---------------------------------------------------------
    p.add_argument("--epochs", type=int, default=1000, help="Upper bound; early stopping usually ends sooner.")
    p.add_argument("--batch_size", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate (AdamW).")
    p.add_argument("--final_lr", type=float, default=1e-4, help="Learning rate at the end of the cosine decay.")
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--lr_decay_epochs", type=int, default=60, help="Cosine decay reaches --final_lr at this epoch.")
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--patience", type=int, default=25, help="Epochs without validation-MAE improvement before stopping.")
    p.add_argument("--min_delta", type=float, default=1e-5, help="Minimum MAE improvement (normalised units) that counts.")
    p.add_argument("--keep_zero_chm", action="store_true",
                   help="Also supervise pixels whose CHM is exactly 0. By default they are excluded "
                        "from the loss, because 0 is the LiDAR no-data value.")
    p.add_argument("--zero_min_fraction", type=float, default=0.05,
                   help="Only exclude zeros in patches where they cover at least this fraction "
                        "(smaller amounts are real open ground).")
    p.add_argument("--init_from_checkpoint", default=None,
                   help="Initialise the weights from another run (fine-tuning). Optimiser and "
                        "epoch counter start from zero; ignored when --output_dir already has last.pth.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rgb_bands", type=int, nargs=3, default=[2, 1, 0],
                   help="Indices of the S2 bands shown as RGB in the figures (B4, B3, B2 for a B2..B12 stack).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def collect_samples(args, sensors, log):
    """One dict per patch with the path of every modality, paired by file stem."""
    s2_files = sorted(glob.glob(os.path.join(args.s2_dir, "*.npy")))
    samples, skipped = [], 0
    for f in s2_files:
        stem = Path(f).stem
        paths = {"S2": f, "CHM": os.path.join(args.chm_dir, stem + ".npy")}
        for s in sensors:
            paths[s["name"]] = os.path.join(s["dir"], stem + ".npy")
        if all(os.path.isfile(v) for v in paths.values()):
            samples.append(paths)
        else:
            skipped += 1
    if skipped:
        log.warning(f"{skipped} Sentinel-2 patches skipped: missing CHM or sensor file with the same stem")
    if not samples:
        raise SystemExit("no complete samples found; check the folders and file names")
    return samples


def compute_min_max_per_band(files, name):
    """Absolute per-band min/max over the training patches, ignoring NaN/Inf."""
    mins = maxs = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN bands
        for f in tqdm(files, desc=f"{name} min/max", leave=False):
            arr = np.load(f).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[None]
            arr = np.where(np.isfinite(arr), arr, np.nan)
            bmin, bmax = np.nanmin(arr, axis=(1, 2)), np.nanmax(arr, axis=(1, 2))
            mins = bmin if mins is None else np.fmin(mins, bmin)
            maxs = bmax if maxs is None else np.fmax(maxs, bmax)
    maxs = np.where(maxs <= mins, mins + 1e-6, maxs)  # constant band guard
    return [float(v) for v in mins], [float(v) for v in maxs]


def random_flip_rotate(mods, chm):
    """Same random flip / rot90 applied to every modality (each on its own grid)."""
    if random.random() < 0.5:
        mods, chm = [m[:, :, ::-1] for m in mods], chm[:, ::-1]
    if random.random() < 0.5:
        mods, chm = [m[:, ::-1, :] for m in mods], chm[::-1, :]
    k = random.randint(0, 3)
    if k:
        mods, chm = [np.rot90(m, k, axes=(1, 2)) for m in mods], np.rot90(chm, k)
    return mods, chm


class PatchDataset(Dataset):
    """Returns ``([s2, sensor_1, ...], chm)`` normalised to [0, 1]."""

    def __init__(self, samples, config, augment=False):
        self.samples = samples
        self.config = config
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        item = self.samples[i]
        mods = [normalize_s2(np.load(item["S2"]), self.config["s2"]["divisor"])]
        for s in self.config["sensors"]:
            mods.append(normalize_sensor(np.load(item[s["name"]]), s["min_per_band"], s["max_per_band"]))
        chm = np.load(item["CHM"]).astype(np.float32)
        if chm.ndim == 3:
            chm = chm[0]
        chm = np.clip(np.nan_to_num(chm, nan=0.0) / self.config["chm_max"], 0.0, 1.0)
        if self.augment:
            mods, chm = random_flip_rotate(mods, chm)
        return ([torch.from_numpy(np.ascontiguousarray(m)) for m in mods],
                torch.from_numpy(np.ascontiguousarray(chm)))


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------
def valid_mask(chm, ignore_zero, min_fraction):
    """1 = pixel enters the loss; 0 = LiDAR no-data (CHM == 0) to be ignored.

    Zeros are only excluded in patches where they cover at least
    ``min_fraction`` of the pixels: a few zeros are real open ground, a large
    block of zeros is a hole in the LiDAR coverage.
    """
    if not ignore_zero:
        return torch.ones_like(chm)
    zero = chm <= 0
    if min_fraction > 0:
        frac = zero.flatten(1).float().mean(dim=1)
        zero = zero & (frac >= min_fraction).view(-1, 1, 1)
    return (~zero).float()


def masked_mae(pred, target, valid):
    return (torch.abs(pred - target) * valid).sum() / valid.sum().clamp_min(1.0)


class AverageMeter:
    def __init__(self):
        self.sum, self.count = 0.0, 0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _stretch(img):
    """Percentile (2-98) contrast stretch to [0, 1]."""
    img = np.nan_to_num(np.asarray(img, dtype=np.float32))
    lo, hi = np.percentile(img, (2, 98))
    return np.clip((img - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(img)


def plot_samples(mods, names, chm, pred, path, rgb_bands, n=4):
    """Rows: S2 RGB, each auxiliary sensor (first band), reference CHM, prediction."""
    n = min(n, chm.shape[0])
    mods = [m[:n].cpu().numpy() for m in mods]
    chm, pred = chm[:n].cpu().numpy(), pred[:n].cpu().numpy()
    rows = [("Sentinel-2", [np.dstack([_stretch(m[b]) for b in rgb_bands]) for m in mods[0]])]
    rows += [(name, [_stretch(m[0]) for m in mod]) for name, mod in zip(names[1:], mods[1:])]
    rows += [("Reference CHM", list(chm)), ("Predicted CHM", list(pred))]

    fig, axes = plt.subplots(len(rows), n, figsize=(3 * n, 2.6 * len(rows)), squeeze=False)
    for r, (label, images) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            ax.set_axis_off()
            if "CHM" in label:
                ax.imshow(images[c], cmap="magma", vmin=0, vmax=max(chm[c].max(), 1e-3))
            else:
                ax.imshow(images[c], cmap=None if images[c].ndim == 3 else "gray")
            if c == 0:
                ax.text(-0.08, 0.5, label, rotation=90, va="center", ha="center",
                        transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, lr_schedule, epoch, args, device, log, fig_path):
    model.train()
    loss_meter = AverageMeter()
    for it, (mods, chm) in enumerate(loader):
        for group in optimizer.param_groups:
            group["lr"] = lr_schedule[epoch * len(loader) + it]
        mods = [m.to(device, non_blocking=True) for m in mods]
        chm = chm.to(device, non_blocking=True)

        pred = torch.sigmoid(model(mods))[:, 0]
        valid = valid_mask(chm, not args.keep_zero_chm, args.zero_min_fraction)
        loss = masked_mae(pred, chm, valid)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), chm.size(0))

        if it % 10 == 0:
            gates = model.gate_stats()
            gate_msg = " | gates " + ", ".join(f"{n}={g:.3f}" for n, g in zip(model.modality_names, gates)) \
                if gates else ""
            log.info(f"epoch {epoch} [{it}/{len(loader)}] loss {loss.item():.4f} "
                     f"(avg {loss_meter.avg:.4f}) | valid {valid.mean().item():.3f} "
                     f"| lr {optimizer.param_groups[0]['lr']:.2e}{gate_msg}")
        if it == 0:
            plot_samples(mods, model.modality_names, chm, pred.detach(), fig_path, args.rgb_bands)
    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, epoch, args, device, log, fig_path):
    model.eval()
    mae_meter = AverageMeter()
    for it, (mods, chm) in enumerate(loader):
        mods = [m.to(device, non_blocking=True) for m in mods]
        chm = chm.to(device, non_blocking=True)
        pred = torch.sigmoid(model(mods))[:, 0]
        valid = valid_mask(chm, not args.keep_zero_chm, args.zero_min_fraction)
        mae_meter.update(masked_mae(pred, chm, valid).item(), chm.size(0))
        if it == 0:
            plot_samples(mods, model.modality_names, chm, pred, fig_path, args.rgb_bands)
    log.info(f"validation epoch {epoch}: MAE {mae_meter.avg:.5f} "
             f"({mae_meter.avg * args.chm_max:.2f} m)")
    return mae_meter.avg


def build_lr_schedule(args, iters_per_epoch):
    """Per-iteration LR: linear warm-up, cosine decay to --final_lr at
    --lr_decay_epochs, then constant --final_lr until --epochs."""
    if args.lr_decay_epochs <= args.warmup_epochs:
        raise SystemExit("--lr_decay_epochs must be larger than --warmup_epochs")
    warmup = np.linspace(0.0, args.lr, iters_per_epoch * args.warmup_epochs)
    n_cos = iters_per_epoch * (args.lr_decay_epochs - args.warmup_epochs)
    cosine = np.array([args.final_lr + 0.5 * (args.lr - args.final_lr) * (1 + math.cos(math.pi * t / n_cos))
                       for t in range(n_cos)])
    tail = np.full(iters_per_epoch * max(0, args.epochs - args.lr_decay_epochs), args.final_lr)
    return np.concatenate([warmup, cosine, tail])


def setup_logger(output_dir):
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(os.path.join(output_dir, "train.log"))):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    log = setup_logger(args.output_dir)
    log.info("arguments: " + json.dumps(vars(args), indent=None))
    set_seed(args.seed)
    device = torch.device(args.device)

    # -- which auxiliary branches are active? ---------------------------------
    sensors = [dict(name=name, dir=getattr(args, dir_attr), resolution_m=getattr(args, res_attr))
               for name, dir_attr, res_attr in SENSORS if getattr(args, dir_attr)]
    log.info("branches: S2" + "".join(f" + {s['name']}" for s in sensors))

    # -- pair files and split ---------------------------------------------------
    samples = collect_samples(args, sensors, log)
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_val = max(1, round(len(samples) * args.val_fraction))
    val_samples, train_samples = samples[:n_val], samples[n_val:]
    log.info(f"{len(train_samples)} training / {len(val_samples)} validation patches")

    # -- reference grid and per-sensor geometry / normalisation ---------------
    s2_probe = np.load(train_samples[0]["S2"])
    s2_channels, patch_size = int(s2_probe.shape[0]), int(s2_probe.shape[-1])
    log.info(f"S2: {s2_channels} bands, {patch_size}x{patch_size} px @ {args.s2_resolution:g} m")
    for s in sensors:
        probe = np.load(train_samples[0][s["name"]])
        probe = probe[None] if probe.ndim == 2 else probe
        s["channels"], s["patch_side"] = int(probe.shape[0]), int(probe.shape[-1])
        s["base_channels"] = args.sensor_base_channels
        s["depth"] = auto_backbone_depth(s["patch_side"])
        expected = round(patch_size * args.s2_resolution / s["resolution_m"])
        if abs(expected - s["patch_side"]) > 1:
            log.warning(f"{s['name']}: patch side is {s['patch_side']} px but {args.s2_resolution:g} m / "
                        f"{s['resolution_m']:g} m resolutions imply {expected} px; check the resolutions")
        if s["name"] == "DEM":
            s["min_per_band"], s["max_per_band"] = [0.0] * s["channels"], [args.dem_divisor] * s["channels"]
        else:
            s["min_per_band"], s["max_per_band"] = compute_min_max_per_band(
                [t[s["name"]] for t in train_samples], s["name"])
        log.info(f"{s['name']}: {s['channels']} bands, {s['patch_side']} px @ {s['resolution_m']:g} m, "
                 f"backbone depth {s['depth']}, min {np.round(s['min_per_band'], 3).tolist()}, "
                 f"max {np.round(s['max_per_band'], 3).tolist()}")

    config = dict(
        encoder_name=args.encoder_name,
        feature_channels=args.feature_channels,
        head_channels=args.head_channels,
        align_mode="bilinear",
        patch_size=patch_size,
        s2=dict(channels=s2_channels, resolution_m=args.s2_resolution, divisor=args.s2_divisor),
        chm_max=args.chm_max,
        sensors=[{k: s[k] for k in SENSOR_CONFIG_KEYS} for s in sensors],
    )
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    train_loader = DataLoader(PatchDataset(train_samples, config, augment=args.augment),
                              batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(PatchDataset(val_samples, config), batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    if len(train_loader) == 0:
        raise SystemExit("fewer training patches than --batch_size")

    # -- model / optimiser ------------------------------------------------------
    model = build_model(config, pretrained=not args.no_pretrained).to(device)
    log.info(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_schedule = build_lr_schedule(args, len(train_loader))
    torch.backends.cudnn.benchmark = True

    start_epoch, best_mae, bad_epochs = 0, float("inf"), 0
    last_path = os.path.join(args.output_dir, "last.pth")
    if os.path.isfile(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, best_mae, bad_epochs = ck["epoch"], ck["best_mae"], ck["bad_epochs"]
        log.info(f"resumed from {last_path} at epoch {start_epoch} (best MAE {best_mae:.5f})")
    elif args.init_from_checkpoint:
        ck = torch.load(args.init_from_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        log.info(f"weights initialised from {args.init_from_checkpoint} (fine-tuning)")

    # -- training loop ------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        log.info(f"===== epoch {epoch} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, lr_schedule, epoch, args, device,
                                     log, os.path.join(figures_dir, f"train_epoch_{epoch}.png"))
        val_mae = validate(model, val_loader, epoch, args, device, log,
                           os.path.join(figures_dir, f"val_epoch_{epoch}.png"))

        improved = (best_mae - val_mae) > args.min_delta
        if improved:
            best_mae, bad_epochs = val_mae, 0
        else:
            bad_epochs += 1
        state = dict(epoch=epoch + 1, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                     best_mae=best_mae, bad_epochs=bad_epochs, train_loss=train_loss,
                     val_mae=val_mae, config=config, args=vars(args))
        torch.save(state, last_path)
        if improved:
            torch.save(state, os.path.join(args.output_dir, "best.pth"))
            log.info(f"new best model (MAE {val_mae:.5f} = {val_mae * args.chm_max:.2f} m)")
        else:
            log.info(f"no improvement for {bad_epochs}/{args.patience} epochs "
                     f"(best {best_mae:.5f}, current {val_mae:.5f})")
        if bad_epochs >= args.patience:
            log.info(f"early stopping at epoch {epoch}")
            break
    log.info(f"done. best validation MAE {best_mae:.5f} ({best_mae * args.chm_max:.2f} m); "
             f"checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main()
