# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-stage pipeline for converting 2D rasterized floor plan images into 3D architectural models. The system combines deep learning-based recognition with classical computer vision to extract walls, doors, windows, and other architectural features, then generates 3D geometry for Blender visualization.

### Three Main Components

1. **Raster-to-Graph/** - Deep learning model that converts floor plan images to vectorized graphs
2. **floor/** - Feature extraction and architectural analysis pipeline
3. **blender/** - 3D geometry generation from extracted features

## Data Flow Pipeline

```
Floorplan Image (JPG)
    → Raster-to-Graph (DL inference) → Graph structure (junctions + edges)
    → floor/ (Feature extraction) → JSON with walls/openings/junctions
    → blender/ (3D generation) → OBJ/MTL files + renders
```

## Component 1: Raster-to-Graph

**Purpose**: Neural network that converts rasterized floor plan images into vectorized graph representations.

### Environment Setup
```bash
cd Raster-to-Graph
conda create -n R2G python=3.7.13
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# Install deformable attention module
cd models/ops/
sh make.sh
cd ../..
```

### Key Commands
```bash
# Training (requires dataset)
python train.py

# Important: At epoch 80, manually stop and adjust learning rates in args.py:
# Change lr=2e-4 lr_backbone=2e-5 lr_linear_proj=2e-5 to (2e-5 2e-6 2e-6)
# Set 'resume' to path/to/79th_checkpoint, then rerun

# Testing
python test.py

# Demo inference on custom images
python demo.py  # Edit demo.py to change input folder
```

### Architecture
- **Model**: Deformable DETR-based transformer with multi-scale attention
- **Backbone**: ResNet50 for visual feature extraction
- **Output**: Graph with junction coordinates and wall segments
- **Core files**:
  - [models/deformable_transformer.py](Raster-to-Graph/models/deformable_transformer.py) - Encoder/decoder architecture
  - [engine.py](Raster-to-Graph/engine.py) - Training/evaluation loops
  - [util/graph_utils.py](Raster-to-Graph/util/graph_utils.py) - Graph construction

## Component 2: floor/

**Purpose**: Analyzes floor plans and extracts architectural features (walls, doors, windows, rooms).

### Environment Setup
```bash
cd floor
pip install -r requirements.txt
```

### Key Pipeline Commands
```bash
# Run complete pipeline (4 stages)
./run_pipeline.sh
# or
python run_pipeline.py
```

**Pipeline stages**:
1. Hatching detection (`hatching_mask.py`) → `enhanced_hatching_strict_mask.png`
2. Object export (`export_objects.py`) → `plan_floor1_objects.json`
3. Opening alignment (`visualize_polygons_align.py`) → aligned openings in JSON
4. Visualization (`visualize_polygons_w.py`) → `wall_coordinates.json` + `wall_polygons.svg`

### Individual Stage Commands
```bash
# Run stages independently for debugging
python hatching_mask.py                    # Wall detection via hatching patterns
python export_objects.py                   # Extract walls/doors/windows to JSON
python visualize_polygons_align.py         # Align openings on walls
python visualize_polygons_w.py             # Generate final visualization
```

### CubiCasa5K Model (used for door/window detection)

The floor analysis uses CubiCasa5K multi-task model for detecting doors and windows.

**Training**:
```bash
python train.py --arch hg_furukawa_original --batch-size 26 --image-size 256
```

**Evaluation**:
```bash
python eval.py --weights model_best_val_loss_var.pkl
```

**Key parameters**:
- Model weights: `model_best_val_loss_var.pkl` (download from Google Drive, see floor/docs/README.md)
- Output: 44 channels (21 heatmaps + 12 room classes + 11 icon classes)
- Uses multi-task uncertainty loss

### Architecture
- **Core module**: `floortrans/` - Multi-task CNN (hourglass architecture)
- **Key scripts**:
  - [cubicasa_vectorize.py](floor/cubicasa_vectorize.py) - Multi-scale DL inference for openings (~3000 lines)
  - [export_objects.py](floor/export_objects.py) - Main extraction script (~4000 lines)
  - [junction_type_analyzer.py](floor/junction_type_analyzer.py) - Junction classification
  - [visualize_polygons_align.py](floor/visualize_polygons_align.py) - Opening alignment (solves J18-J25 issue)

### Important Notes
- **Opening alignment** is critical: normalizes thickness and aligns doors/windows to avoid wall fragmentation
- Junction analysis determines wall connectivity (T-junctions, corners, etc.)
- See [floor/CLAUDE.md](floor/CLAUDE.md) for detailed CubiCasa5K documentation

## Component 3: blender/

**Purpose**: Converts 2D wall coordinates to 3D geometry with proper normals for Blender rendering.

### Key Commands
```bash
cd blender

# Main 3D generation (run in Blender)
blender --python create_walls_2m.py

# Or using integration script
blender --python run_final_integration.py
```

### Output Files
- `wall_coordinates_3d.obj` - 3D wall mesh
- `wall_coordinates_3d.mtl` - Material definitions
- `wall_coordinates_isometric.jpg` - Isometric render
- `external_normals.json` - Normal vectors for wall exterior surfaces

### Architecture
- **Main script**: [create_walls_2m.py](blender/create_walls_2m.py) (~180KB) - Core 3D generation logic
  - Loads wall coordinates from JSON
  - Creates 3D meshes with wall thickness (2m default)
  - Calculates external surface normals
  - Handles complex multi-segment junctions
  - Exports OBJ/MTL files

- **Supporting modules**:
  - [enhanced_universal_segmentation.py](blender/enhanced_universal_segmentation.py) - Wall segment grouping
  - [enhanced_wall_grouping.py](blender/enhanced_wall_grouping.py) - Logical wall grouping
  - [wall_grouping_functions.py](blender/wall_grouping_functions.py) - Segment analysis helpers

### Key Concepts

**External Normals Problem**:
Multi-segment walls (e.g., wall from J6→J4 with windows) can have inconsistent normals across segments, creating a "checkerboard" pattern. The system now ensures all segments of the same physical wall share the same normal vector.

**Wall Segment Naming**:
- Format: `wall_window_X_top_Y_to_Z` or `wall_window_X_bottom_Y_to_Z`
- X = window/opening number
- Y, Z = junction IDs

**Coordinate System**:
- Blender uses Y-up coordinate system
- JSON coordinates may need inversion (see `INVERT_COORDINATES_FIXED_README.md`)

### Debug Scripts
```bash
python debug_wall_sequences.py              # Trace wall connectivity
python debug_wall_6_segments.py             # Analyze multi-segment junctions
python analyze_multi_segment_junctions.py   # Junction topology analysis
python compare_normals.py                   # Compare normal calculation methods
```

## Data Structure: wall_coordinates.json

Central data format used between floor/ and blender/:

```json
{
  "walls": [
    {
      "id": "wall_1",
      "start_junction": "J1",
      "end_junction": "J2",
      "coordinates": [[x1, y1], [x2, y2]],
      "thickness": 0.2,
      "segments": [...]
    }
  ],
  "junctions": [
    {
      "id": "J1",
      "position": [x, y],
      "type": "corner|T-junction|4-way",
      "connected_walls": ["wall_1", "wall_2"]
    }
  ],
  "openings": [
    {
      "id": "window_1",
      "type": "window|door",
      "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "parent_wall": "wall_1"
    }
  ]
}
```

## Common Workflows

### Complete Pipeline: Image → 3D Model
```bash
# 1. (Optional) Generate graph from image
cd Raster-to-Graph
python demo.py  # Configure input in demo.py

# 2. Extract features
cd ../floor
./run_pipeline.sh

# 3. Generate 3D model
cd ../blender
blender --python create_walls_2m.py
```

### Debug Pipeline Stage Issues
```bash
# Check hatching detection
cd floor
python hatching_mask.py
# Inspect: enhanced_hatching_strict_mask.png

# Check object extraction
python export_objects.py
# Inspect: plan_floor1_objects.json

# Check alignment
python visualize_polygons_align.py
# Compare before/after in JSON

# Check visualization
python visualize_polygons_w.py
# Inspect: wall_polygons.svg
```

### Test Normal Calculation
```bash
cd blender
python test_external_normals_fixed.py
# Verifies consistent normals for multi-segment walls
```

## Key Architectural Patterns

1. **Cascade Processing**: Each stage builds on previous outputs
   - Raster-to-Graph provides structural foundation
   - floor/ extracts features from structure
   - blender/ converts to 3D

2. **Graph-Based Representation**: Floorplans modeled as graphs (junctions=nodes, walls=edges)

3. **Multi-Task Learning**: CubiCasa5K predicts heatmaps + rooms + icons simultaneously

4. **JSON-Based Exchange**: Decoupled components communicate via standardized JSON

5. **Iterative Refinement**: Pipeline includes alignment step to correct DL predictions

## Important Files to Read First

When modifying the pipeline, start with:
- [floor/docs/PIPELINE_README.md](floor/docs/PIPELINE_README.md) - Complete pipeline documentation
- [blender/README_FINAL_INTEGRATION.md](blender/README_FINAL_INTEGRATION.md) - 3D generation details
- [floor/docs/README_wall_alignment.md](floor/docs/README_wall_alignment.md) - Opening alignment logic
- [floor/CLAUDE.md](floor/CLAUDE.md) - CubiCasa5K model details

## Known Issues

- Raster-to-Graph requires manual learning rate adjustment at epoch 80
- CubiCasa5K LMDB database is 105GB (uses float32 instead of uint8)
- Augmentations in floortrans/ operate on torch tensors (should use numpy)
- Wall normal calculation for multi-segment walls required special handling (now fixed)
- Coordinate inversion between 2D and 3D coordinate systems must be handled carefully
