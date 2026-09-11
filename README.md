# RCAFNet

RCAFNet is short for **Recalibrated Cross-Attention Fusion Network**. It is an
RGB-event semantic segmentation network that first mutually recalibrates image
and event features with a Cross Feature Recalibration module, then fuses the two
modalities with a Cross-Attention Fusion module.

The implementation is built around a dual-stream MiT encoder. RGB and event
features are extracted by separate MiT backbones, recalibrated stage by stage,
fused by cross-attention, and decoded by an MLP-style SegFormer head.

The repository also keeps an EISNet-style baseline in the same training
framework for comparison.

## Project Structure

```text
RCAFNet/
  configs/                    # Complete experiment configs
    rcafnet_dsec.yaml
    rcafnet_ddd17.yaml
    eisnet_dsec.yaml
    eisnet_ddd17.yaml
  tools/
    train.py                  # Main training entry point
    precompute_dsec_events.py # Optional DSEC event-frame cache builder
    complexity.py             # Parameter, MAC/FLOP and latency profiler
    visualize_segmentation.py # Save RGB/event/prediction/label visualizations
    visualize_attention.py    # Save CrossFE/CAFFM attention/debug maps
    evaluate.py               # Placeholder
    infer.py                  # Placeholder
  src/
    config/                   # YAML loading helpers
    datasets/                 # CARLA, DSEC, DDD17 datasets and transforms
    engine/                   # Training, validation, checkpoints, experiments
    losses/                   # CE and CE + Lovasz losses
    metrics/                  # Semantic segmentation metrics
    models/                   # MiT backbones, encoders, fusion modules, decoders
    utils/                    # Device, seed and visualization helpers
  outputs/
    logs/                     # Training runs
    complexity/               # Optional profiler output
```

## Supported Models

Naming note: some source files and class names still use historical implementation names such as `carefnet` or `crossrecalibnet`. The public model registry is now normalized to `rcafnet` and `esinet`.

Registered model names:

- `rcafnet`: Recalibrated Cross-Attention Fusion Network.
- `esinet`: EISNet-style AEIM + MRFM baseline.

The main RCAFNet path uses:

- RGB/Event dual-stream MiT backbones.
- `10c` event representation, or `aet` for EISNet configs.
- Cross Feature Recalibration to mutually calibrate image and event features.
- Cross-Attention Fusion to integrate the recalibrated RGB/event features.
- MLP decoder head by default.

## Environment

There is no pinned `requirements.txt` in this folder. The imports indicate the
following core dependencies:

```bash
pip install torch torchvision timm numpy pillow matplotlib tqdm pyyaml omegaconf tensorboard
```

Use a CUDA-enabled PyTorch build if training on GPU.

## Configs

| Config | Model | Dataset | Event frame | Image size | Classes |
| --- | --- | --- | --- | --- | --- |
| `configs/rcafnet_dsec.yaml` | `rcafnet` | DSEC | `10c` | `640x440` | 11 |
| `configs/rcafnet_ddd17.yaml` | `rcafnet` | DDD17 | `10c` | `346x200` | 6 |
| `configs/eisnet_dsec.yaml` | `esinet` | DSEC | `aet` | `640x440` | 11 |
| `configs/eisnet_ddd17.yaml` | `esinet` | DDD17 | `aet` | `346x200` | 6 |

Before running, edit the config paths for:

- `data_root`
- `encoder.rgb_pretrained`
- `encoder.event_pretrained`
- `precomputed_event_root`, when using precomputed DSEC events

Note: `tools/train.py` now defaults to `configs/rcafnet_dsec.yaml`. Passing `--config` explicitly is still recommended for reproducible experiments.

## Data Layout

### DSEC

Expected layout:

```text
<data_root>/
  image/<sequence>/evt_inf/*.png
  event/<sequence>/data/<frame_id>.npz
  label/<sequence>/11classes/<frame_id>.png
```

If `use_precomputed_events: true`, cached event frames are loaded from:

```text
<precomputed_event_root>/
  metadata.json
  <sequence>/data/<frame_id>.npy
```

### DDD17

Expected layout:

```text
<data_root>/
  dir*/imgs/img_00000002.png
  dir*/imgs/0000000002.png
  dir*/segmentation_masks/segmentation_00000002.png
  dir*/index/index_<t_interval>ms.npy
```

DDD17 uses grayscale images by default. The local split is controlled in the
config with `train_dirs`, `val_dirs`, and `test_dirs`.

### CARLA

CARLA is supported by dataset code, although no CARLA config is currently
included. The dataset expects RGB, event, and semantic folders inside each town
sequence.

## Training

Run commands from the `RCAFNet` directory:

```bash
cd RCAFNet
python tools/train.py --config configs/rcafnet_dsec.yaml --tag rcafnet_dsec --device cuda:0
```

DDD17:

```bash
python tools/train.py --config configs/rcafnet_ddd17.yaml --tag rcafnet_ddd17 --device cuda:0
```

EISNet baseline:

```bash
python tools/train.py --config configs/eisnet_dsec.yaml --tag eisnet_dsec --device cuda:0
python tools/train.py --config configs/eisnet_ddd17.yaml --tag eisnet_ddd17 --device cuda:0
```

Training creates a timestamped run under `outputs/logs/`, for example:

```text
outputs/logs/26-06-29--20-35-00_rcafnet_dsec_dsec_rcafnet/
  config.yaml
  best.pth
  latest.pth
  best_val_metrics.txt
  events.out.tfevents.*
```

Resume a run:

```bash
python tools/train.py --config configs/rcafnet_dsec.yaml --resume outputs/logs/<run_dir> --device cuda:0
```

## DSEC Event Precomputation

For DSEC `10c` configs, precomputing event frames avoids rebuilding event
representations during each training epoch.

```bash
python tools/precompute_dsec_events.py --config configs/rcafnet_dsec.yaml --split all
```

Useful options:

```bash
python tools/precompute_dsec_events.py --config configs/rcafnet_dsec.yaml --split train --limit 100
python tools/precompute_dsec_events.py --config configs/rcafnet_dsec.yaml --output-root /path/to/cache --dtype float16 --overwrite
```

After precomputation, keep these config fields aligned with the generated cache:

```yaml
use_precomputed_events: true
precomputed_event_root: /path/to/cache
```

## Visualization

Visualize semantic predictions from a run:

```bash
python tools/visualize_segmentation.py --run_dir outputs/logs/<run_dir> --random --num_samples 10 --device cuda:0
```

Or pass config and checkpoint manually:

```bash
python tools/visualize_segmentation.py --config configs/rcafnet_dsec.yaml --checkpoint outputs/logs/<run_dir>/best.pth --output_dir outputs/segmentation_vis --device cuda:0
```

Visualize Cross Feature Recalibration and CAFFM debug maps:

```bash
python tools/visualize_attention.py --run_dir outputs/logs/<run_dir> --num_samples 20 --stages 1 2 3 4 --device cuda:0
```

The attention tool requires a model whose encoder exposes
`set_visualization()`, such as the RCAFNet encoder variant.

## Complexity Profiling

```bash
python tools/complexity.py --config configs/rcafnet_dsec.yaml --device cuda:0 --batch_size 1 --repeats 50 --output outputs/complexity/rcafnet_dsec.json
```

The profiler reports parameters, trainable parameters, MACs, estimated FLOPs,
optional latency, per-module MAC totals, and the most expensive modules. MACs
cover Conv2d, Linear, and the custom cross-attention matrix multiplications.

## Metrics and Losses

Training and validation report:

- Per-class IoU
- mIoU
- Pixel accuracy
- Mean class accuracy
- Validation loss

Supported losses:

- `ce`: cross entropy with `ignore_index` and optional label smoothing.
- `ce_lovasz`: cross entropy plus Lovasz softmax.

## Current Limitations

- `tools/evaluate.py` is not implemented yet.
- `tools/infer.py` is not implemented yet.
- No pinned dependency file is included.
- Several config paths are machine-local absolute paths and should be edited
  before training on a new machine.




