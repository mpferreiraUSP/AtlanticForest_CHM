# Multi-sensor canopy height estimation

Deep-learning estimation of canopy height (CHM, in metres) from **Sentinel-2**
optical imagery fused with **Sentinel-1** (C-band SAR), **ALOS PALSAR** (L-band
SAR) and a **digital elevation model**, each ingested at its native resolution.
The network is trained against LiDAR-derived canopy height models and predicts
a 10 m CHM on the Sentinel-2 grid.

The repository contains the training and inference code only; patch
extraction and evaluation are left to the user (see
[Data preparation](#data-preparation) for the expected format).

## How it works

```
Sentinel-2  (10 bands @ 10 m, 256 px) --> U-Net, ResNet-50 encoder (ImageNet) --\
ALOS PALSAR (HH, HV  @ 25 m, 102 px) --> light U-Net                           --\  gated     regression
Sentinel-1  (VV, VH  @ 10 m, 256 px) --> light U-Net                           --/  fusion --> head --> CHM (m)
DEM         (1 band  @ 30 m,  85 px) --> light U-Net                           --/
```

* **One branch per sensor.** The Sentinel-2 branch is a U-Net with a
  ResNet-50 encoder (4 stages, scSE attention, from
  [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)).
  Every auxiliary sensor gets a light U-Net whose depth is chosen
  automatically from the patch size, so a 102 px ALOS patch is not
  down-sampled into nothing. Each branch outputs 64 feature maps.
* **No input resampling.** Every patch enters the network on its own grid.
  The feature maps of the auxiliary branches are bilinearly resampled to the
  Sentinel-2 grid *inside* the network, right before fusion.
* **Gated fusion.** A global descriptor of all branches drives a small MLP
  that produces one gate per channel and per sensor; the gated features are
  concatenated and refined by a 3x3 convolution. The gates are logged during
  training, which tells you how much each sensor is actually being used.
* **Regression head.** Two 3x3 convolutions and a 1x1 output. A sigmoid maps
  the output to [0, 1]; multiplying by `--chm_max` (default 100 m) gives
  metres.
* **Loss.** Mean absolute error. Pixels whose reference CHM is exactly 0 are
  excluded from the loss when they cover at least 5 % of the patch, because 0
  is the LiDAR no-data value (a few isolated zeros are real open ground and
  are kept).

Any subset of the auxiliary sensors can be used: a branch exists only if its
folder is passed to `train.py`, so the same code trains the S2-only,
S2 + S1, S2 + S1 + ALOS, ... ablations.

## Repository layout

| File | Purpose |
|------|---------|
| `model.py` | Network definition, normalisation helpers, checkpoint loading. Shared by the two scripts. |
| `train.py` | Pairs the patches, computes the normalisation statistics, trains with early stopping, writes `best.pth` / `last.pth`. |
| `predict.py` | Rebuilds the network from a checkpoint and writes one CHM raster per input patch. |
| `requirements.txt` | Python dependencies. |

## Installation

```bash
git clone <this repository>
cd <this repository>
pip install -r requirements.txt
```

PyTorch with CUDA support is strongly recommended for training (the full
model has ~45 M parameters). Follow the [PyTorch instructions](https://pytorch.org/get-started/locally/)
for your platform if `pip install torch` does not pick a GPU build.

## Data preparation

Patches are stored as one file per patch and per modality, **paired by file
name**: `S2/patch_0001.npy`, `CHM/patch_0001.npy`, `S1/patch_0001.npy`, ...

```
data/
  train/
    S2/    patch_0001.npy  (C, 256, 256)   Sentinel-2 reflectance x 10000
    CHM/   patch_0001.npy  (256, 256)      LiDAR canopy height in metres, 0 = no data
    S1/    patch_0001.npy  (2, 256, 256)   Sentinel-1 VV, VH (dB)            optional
    ALOS/  patch_0001.npy  (2, 102, 102)   ALOS PALSAR HH, HV (dB)           optional
    DEM/   patch_0001.npy  (1, 85, 85)     elevation in metres               optional
  test/
    S2/    patch_0101.tif  ...             GeoTIFF (or .npy) with the same layout
    S1/, ALOS/, DEM/
```

* Training patches are `.npy` arrays. Inference patches can be GeoTIFF (the
  output is then a georeferenced GeoTIFF) or `.npy`.
* All patches of one sample cover the **same ground footprint**, each at the
  sensor's native resolution: `side = 256 x 10 m / sensor resolution`
  (ALOS at 25 m -> 102 px, DEM at 30 m -> 85 px). `train.py` warns when a
  folder does not match the resolutions passed on the command line.
* Sentinel-2 bands are divided by `--s2_divisor` (10000 for reflectance
  stored as integers). The published models use the ten bands
  B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12.
* Sentinel-1 and ALOS patches are scaled per band with the min/max computed
  over the training patches (any unit works, dB is what the published models
  used). The DEM is divided by a fixed `--dem_divisor` (2100 m by default).
  All statistics are stored in the checkpoint, so inference normalises
  exactly like training.
* The train/validation split is a seeded random split of the patch list.
  Make sure the training patches do not overlap, otherwise the validation
  loss is optimistic.

## Training

```bash
# all four sensors
python train.py \
    --s2_dir data/train/S2 --chm_dir data/train/CHM \
    --s1_dir data/train/S1 --alos_dir data/train/ALOS --dem_dir data/train/DEM \
    --output_dir runs/s2_s1_alos_dem

# Sentinel-2 + Sentinel-1 only (just drop the folders you do not have)
python train.py --s2_dir data/train/S2 --chm_dir data/train/CHM \
    --s1_dir data/train/S1 --output_dir runs/s2_s1

# fine-tune a previous run on more data with a smaller learning rate
python train.py ... --init_from_checkpoint runs/s2_s1_alos_dem/best.pth --lr 1e-4
```

`--output_dir` receives:

| File | Content |
|------|---------|
| `best.pth` | weights of the epoch with the lowest validation MAE (use this one) |
| `last.pth` | weights of the last epoch; training resumes from it automatically if you re-run the same command |
| `config.json` | architecture + normalisation statistics (also embedded in the checkpoints) |
| `train.log` | per-iteration loss, learning rate, fusion gates, validation MAE |
| `figures/` | one PNG per epoch with inputs, reference and prediction for a few patches |

Main options (`python train.py --help` lists them all):

| Option | Default | Meaning |
|--------|---------|---------|
| `--epochs` / `--patience` | 1000 / 25 | upper bound on epochs; stop after 25 epochs without validation improvement |
| `--batch_size` | 15 | fits a 12 GB GPU with four branches at 256 px |
| `--lr`, `--final_lr`, `--warmup_epochs`, `--lr_decay_epochs` | 1e-3, 1e-4, 5, 60 | linear warm-up, cosine decay to `--final_lr` at epoch 60, then constant |
| `--chm_max` | 100 | CHM normalisation (metres); predictions saturate at this height |
| `--s2_resolution`, `--s1_resolution`, `--alos_resolution`, `--dem_resolution` | 10, 10, 25, 30 | native resolutions, used to check the patch sizes |
| `--augment` | off | random flips and 90-degree rotations |
| `--keep_zero_chm` | off | also supervise pixels with CHM == 0 |
| `--encoder_name` | resnet50 | any encoder supported by segmentation_models_pytorch |

## Inference

```bash
python predict.py --checkpoint runs/s2_s1_alos_dem/best.pth \
    --s2_dir data/test/S2 --s1_dir data/test/S1 --alos_dir data/test/ALOS --dem_dir data/test/DEM \
    --output_dir runs/s2_s1_alos_dem/predictions
```

* Only the folders of the sensors the model was trained with are needed; the
  checkpoint knows which ones.
* One raster per Sentinel-2 patch is written with the same file stem:
  float32 GeoTIFF in metres (georeferencing copied from the input), or
  `--uint16` for a compact uint16 GeoTIFF scaled by 1000.
* Patches already present in `--output_dir` are skipped (`--overwrite` to
  recompute), so an interrupted run can simply be restarted.
* Patches smaller than the training size are reflect-padded and cropped back;
  larger patches are processed whole (the network is fully convolutional).

## Checkpoints

A checkpoint is a `torch.save` dictionary with `state_dict`, `optimizer`,
`epoch`, `best_mae` and `config`. `config` is all `predict.py` needs:

```json
{
  "encoder_name": "resnet50", "feature_channels": 64, "head_channels": 64,
  "patch_size": 256, "chm_max": 100.0,
  "s2": {"channels": 10, "resolution_m": 10.0, "divisor": 10000.0},
  "sensors": [
    {"name": "ALOS", "channels": 2, "resolution_m": 25.0, "patch_side": 102,
     "base_channels": 32, "depth": 1, "min_per_band": [...], "max_per_band": [...]},
    {"name": "S1", ...}, {"name": "DEM", ...}
  ]
}
```

To use a checkpoint from Python:

```python
from model import load_model
model, config, _ = load_model("runs/s2_s1_alos_dem/best.pth", "cuda")
```

## Citation

If you use this code, please cite:

```bibtex
@article{ferreira_chm_multisensor,
  title   = {<paper title>},
  author  = {Ferreira, Matheus P. and ...},
  journal = {<journal>},
  year    = {<year>},
  doi     = {<doi>}
}
```

## License

<add a license, e.g. MIT or Apache-2.0>

## Contact

Matheus Pinheiro Ferreira -- mpferreira@usp.br
