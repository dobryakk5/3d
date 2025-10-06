# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CubiCasa5K is a deep learning project for floorplan image analysis using a multi-task model. The model performs three simultaneous tasks:
1. **Heatmap prediction** (21 channels) - detecting wall junctions, icon corners, and opening endpoints
2. **Room segmentation** (12 classes) - classifying pixels into room types
3. **Icon segmentation** (11 classes) - detecting architectural elements like doors, windows, toilets, etc.

Based on the "Raster-to-Vector: Revisiting Floorplan Transformation" architecture with multi-task uncertainty loss.

## Environment

- **Python**: 3.6.5
- **PyTorch**: 1.0.0 with CUDA
- **OpenCV**: 3.1.0 (cv2) - install separately, not in requirements.txt
- **Docker**: Optional but recommended. Uses `anibali/pytorch:cuda-9.0` base image

### Docker Setup
```bash
# Build container
docker build -t cubi -f Dockerfile .

# Run JupyterLab
docker run --rm -it --init \
  --runtime=nvidia \
  --ipc=host \
  --publish 1111:1111 \
  --user="$(id -u):$(id -g)" \
  --volume=$PWD:/app \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  cubi jupyter-lab --port 1111 --ip 0.0.0.0 --no-browser
```

## Data Preparation

Dataset should be extracted to `data/cubicasa5k/` directory.

### Create LMDB Database
The model uses LMDB database for faster data loading (~105GB on disk):
```bash
python create_lmdb.py --txt val.txt
python create_lmdb.py --txt test.txt
python create_lmdb.py --txt train.txt
```

Alternative: Use `format='txt'` in data loaders (slower, but no LMDB required).

## Training

```bash
python train.py
```

### Key Training Arguments
- `--arch hg_furukawa_original` - Model architecture (default)
- `--optimizer adam-patience-previous-best` - Optimizer choice
- `--data-path data/cubicasa5k/` - Path to dataset
- `--n-classes 44` - Total output classes (21 heatmaps + 12 rooms + 11 icons)
- `--n-epoch 1000` - Number of epochs
- `--batch-size 26` - Batch size
- `--image-size 256` - Training image size
- `--l-rate 1e-3` - Learning rate
- `--patience 10` - Learning rate scheduler patience
- `--weights <path>` - Resume from checkpoint
- `--new-hyperparams` - Continue training with new hyperparameters
- `--debug` - Debug mode (sets num_workers=0)
- `--plot-samples` - Plot floorplan segmentations to TensorBoard
- `--scale` - Enable rescale augmentation

### TensorBoard
```bash
tensorboard --logdir runs_cubi/
```

Logs are saved to `runs_cubi/<timestamp>/` with:
- `train.log` - Training logs
- `args.json` - Training arguments
- `model_best_val_loss_var.pkl` - Best model by validation loss with variance
- `model_best_val_loss.pkl` - Best model by validation loss
- `model_best_val_acc.pkl` - Best model by pixel accuracy
- `model_last_epoch.pkl` - Final epoch model

## Evaluation

Download pre-trained weights from [here](https://drive.google.com/file/d/1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK/view?usp=sharing) and place in project root.

```bash
python eval.py --weights model_best_val_loss_var.pkl
```

Results are saved to `runs_cubi/<timestamp>/eval.log`.

## Interactive Demo

Use `samples.ipynb` Jupyter notebook to:
- Load pre-trained model
- Visualize predictions on test samples
- Compare ground truth vs predictions
- See post-processed polygon outputs

## Code Architecture

### Module Structure
```
floortrans/
├── loaders/          # Data loading and augmentation
│   ├── house.py      # Room/icon class definitions (~80 classes)
│   ├── svg_loader.py # SVG floorplan parsing
│   ├── svg_utils.py  # Polygon utilities, heatmap generation
│   └── augmentations.py # Data augmentation (operates on torch tensors)
├── models/           # Neural network architectures
│   ├── hg_furukawa_original.py # Hourglass architecture
│   └── model_1427.py
├── losses/           # Loss functions
│   └── uncertainty_loss.py # Multi-task uncertainty loss
├── metrics.py        # Evaluation metrics (IoU, pixel accuracy)
├── plotting.py       # Visualization utilities
└── post_prosessing.py # Polygon extraction from predictions
```

### Key Components

**UncertaintyLoss** ([floortrans/losses/uncertainty_loss.py](floortrans/losses/uncertainty_loss.py))
- Multi-task loss with learnable uncertainty weighting
- Combines cross-entropy (rooms, icons) + MSE (heatmaps)
- `input_slice=[21, 12, 11]` defines task channel splits
- Learnable parameters: `log_vars` (2 params for rooms/icons), `log_vars_mse` (21 params for heatmaps)

**FloorplanSVG Data Loader** ([floortrans/loaders/svg_loader.py](floortrans/loaders/svg_loader.py))
- Supports two formats: `format='lmdb'` (fast) or `format='txt'` (slow, parses SVG on-the-fly)
- Returns dict: `{'image', 'label', 'heatmaps', 'folder'}`
- Labels shape: [3, H, W] → [heatmap_channels, room_class, icon_class]

**Augmentations** ([floortrans/loaders/augmentations.py](floortrans/loaders/augmentations.py))
- Currently operates on torch tensors (should be refactored to numpy)
- `RandomCropToSizeTorch` - Random crop to fixed size
- `ResizePaddedTorch` - Resize with padding
- `RandomRotations` - 90° rotations
- `ColorJitterTorch` - Color augmentation

**Post-processing** ([floortrans/post_prosessing.py](floortrans/post_prosessing.py))
- Converts heatmap predictions to polygon representations
- Uses OpenCV for contour detection and vectorization

### Prediction Format

Model outputs 44 channels split as:
- Channels 0-20: Heatmaps (21 junction/corner types)
- Channels 21-32: Room segmentation logits (12 classes)
- Channels 33-43: Icon segmentation logits (11 classes)

Use `torch.split(output, [21, 12, 11], dim=1)` to separate tasks.

### Room Classes (12 total)
Background, Outdoor, Wall, Kitchen, Living Room, Bedroom, Bath, Entry/Hallway, Railing, Storage, Garage, Other rooms

See [floortrans/loaders/house.py](floortrans/loaders/house.py) for full 80+ category mapping.

### Icon Classes (11 total)
Empty, Window, Door, Closet, Electrical Appliance, Toilet, Sink, Sauna bench, Fire Place, Bathtub, Chimney

## Known Issues / Todo

From README.md:
- LMDB stores float32 instead of uint8 (causes 100GB+ database size)
- Augmentations should operate on numpy arrays instead of torch tensors
