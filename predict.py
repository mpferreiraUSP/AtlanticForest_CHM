#!/usr/bin/env python3
"""
Predict canopy height from patches with a trained checkpoint.

The checkpoint carries everything needed to rebuild the network and normalise
the inputs exactly as in training (``config``); you only point the script at
the patch folders. One output raster per Sentinel-2 patch is written to
``--output_dir`` with the same file stem. GeoTIFF inputs give georeferenced
GeoTIFF outputs (CHM in metres, float32); ``.npy`` inputs give ``.npy`` outputs.

Example::

    python predict.py --checkpoint runs/s2_s1_alos_dem/best.pth \
        --s2_dir data/test/S2 --s1_dir data/test/S1 --alos_dir data/test/ALOS \
        --dem_dir data/test/DEM --output_dir runs/s2_s1_alos_dem/predictions
"""
import argparse
import glob
import math
import os
from pathlib import Path

import numpy as np
import rasterio
import torch

from model import load_model, normalize_s2, normalize_sensor

EXTENSIONS = (".tif", ".tiff", ".TIF", ".TIFF", ".npy")


def parse_args():
    p = argparse.ArgumentParser(description="CHM inference from a trained checkpoint.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="best.pth / last.pth written by train.py.")
    p.add_argument("--s2_dir", required=True, help="Sentinel-2 patches (.tif or .npy).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--s1_dir", default=None, help="Sentinel-1 patches (required if the model uses S1).")
    p.add_argument("--alos_dir", default=None, help="ALOS PALSAR patches (required if the model uses ALOS).")
    p.add_argument("--dem_dir", default=None, help="DEM patches (required if the model uses DEM).")
    p.add_argument("--uint16", action="store_true",
                   help="Write uint16 GeoTIFFs scaled by 1000 (value / 1000 = metres) instead of float32.")
    p.add_argument("--overwrite", action="store_true", help="Recompute patches that already have an output.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def read_patch(path):
    """Return ``(array (C, H, W) float32, rasterio profile or None)``."""
    if path.lower().endswith(".npy"):
        arr, profile = np.load(path), None
    else:
        with rasterio.open(path) as src:
            arr, profile = src.read(), src.profile.copy()
    arr = arr.astype(np.float32)
    return (arr[None] if arr.ndim == 2 else arr), profile


def find_patch(folder, stem):
    for ext in EXTENSIONS:
        candidate = os.path.join(folder, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def pad_reflect(arr, height, width):
    """Centre ``arr`` (C, h, w) inside (C, height, width) with reflection padding."""
    ph, pw = height - arr.shape[1], width - arr.shape[2]
    top, left = ph // 2, pw // 2
    return np.pad(arr, ((0, 0), (top, ph - top), (left, pw - left)), mode="reflect"), (top, left)


@torch.no_grad()
def predict_patch(model, config, s2, sensors, device):
    """``s2`` (C, H, W) raw reflectance; ``sensors`` list of raw arrays in the
    order of ``config["sensors"]``. Returns the CHM in metres, (H, W)."""
    _, h, w = s2.shape
    # The network needs at least the training patch size and sides divisible
    # by 16 (four encoder stages); smaller patches are reflect-padded, and the
    # auxiliary patches are padded by the same proportion to keep alignment.
    th = max(config["patch_size"], 16 * math.ceil(h / 16))
    tw = max(config["patch_size"], 16 * math.ceil(w / 16))
    s2_in, (top, left) = pad_reflect(normalize_s2(s2, config["s2"]["divisor"]), th, tw)
    inputs = [s2_in]
    for arr, spec in zip(sensors, config["sensors"]):
        arr = normalize_sensor(arr, spec["min_per_band"], spec["max_per_band"])
        sh = max(1, round(arr.shape[1] * th / h))
        sw = max(1, round(arr.shape[2] * tw / w))
        inputs.append(pad_reflect(arr, sh, sw)[0])
    tensors = [torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(device) for x in inputs]
    pred = torch.sigmoid(model(tensors))[0, 0].cpu().numpy()
    return pred[top:top + h, left:left + w] * config["chm_max"]


def save_prediction(pred, profile, path, uint16):
    if profile is None:  # .npy input -> .npy output
        np.save(path, pred.astype(np.float32))
        return
    profile.update(count=1, compress="lzw")
    profile.pop("nodata", None)
    if uint16:
        data = np.clip(np.round(pred * 1000.0), 0, 65535).astype(np.uint16)
        profile.update(dtype="uint16")
    else:
        data = pred.astype(np.float32)
        profile.update(dtype="float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        if uint16:
            dst.update_tags(SCALE_FACTOR=1000, DESCRIPTION="canopy height: value / 1000 = metres")


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    model, config, ck = load_model(args.checkpoint, device)
    print(f"checkpoint: {args.checkpoint} (epoch {ck.get('epoch', '?')})")
    print(f"branches: {' + '.join(model.modality_names)} | S2 {config['s2']['channels']} bands, "
          f"patch {config['patch_size']} px | chm_max {config['chm_max']:g} m")

    folders = {"S1": args.s1_dir, "ALOS": args.alos_dir, "DEM": args.dem_dir}
    sensor_dirs = []
    for spec in config["sensors"]:
        folder = folders.get(spec["name"])
        if not folder:
            raise SystemExit(f"the model uses {spec['name']}: pass --{spec['name'].lower()}_dir")
        sensor_dirs.append(folder)

    s2_files = set()
    for ext in EXTENSIONS:
        s2_files.update(glob.glob(os.path.join(args.s2_dir, "*" + ext)))
    s2_files = sorted(s2_files)
    if not s2_files:
        raise SystemExit(f"no patches found in {args.s2_dir}")
    print(f"{len(s2_files)} Sentinel-2 patches in {args.s2_dir}")

    done = skipped = 0
    for i, s2_path in enumerate(s2_files, 1):
        stem = Path(s2_path).stem
        is_npy = s2_path.lower().endswith(".npy")
        out_path = os.path.join(args.output_dir, stem + (".npy" if is_npy else ".tif"))
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        s2, profile = read_patch(s2_path)
        if s2.shape[0] != config["s2"]["channels"]:
            print(f"[{i}/{len(s2_files)}] {stem}: expected {config['s2']['channels']} S2 bands, "
                  f"got {s2.shape[0]} -- skipped")
            skipped += 1
            continue
        sensors, missing = [], None
        for spec, folder in zip(config["sensors"], sensor_dirs):
            path = find_patch(folder, stem)
            if path is None:
                missing = f"{spec['name']} patch not found in {folder}"
                break
            arr, _ = read_patch(path)
            if arr.shape[0] != spec["channels"]:
                missing = f"expected {spec['channels']} {spec['name']} bands, got {arr.shape[0]}"
                break
            sensors.append(arr)
        if missing:
            print(f"[{i}/{len(s2_files)}] {stem}: {missing} -- skipped")
            skipped += 1
            continue

        pred = predict_patch(model, config, s2, sensors, device)
        save_prediction(pred, profile, out_path, args.uint16)
        done += 1
        print(f"[{i}/{len(s2_files)}] {stem}: CHM {pred.min():.1f}-{pred.max():.1f} m -> {os.path.basename(out_path)}")

    print(f"finished: {done} predicted, {skipped} skipped, outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
