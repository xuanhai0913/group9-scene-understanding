---
name: segmentation-depth-implementation
description: Use when implementing or reviewing code/notebooks for semantic segmentation plus depth estimation in the Group 9 traffic scene understanding project. Focuses on Python, OpenCV, NumPy, Matplotlib, PyTorch, U-Net/pretrained segmentation, MiDaS, visualization, and testable outputs.
---

# Segmentation + Depth Implementation

## Implementation workflow

Use this order for notebooks or scripts:

1. Import libraries and set paths.
2. Load a street-scene image.
3. Preprocess: RGB/BGR conversion, resize, normalize.
4. Run semantic segmentation.
5. Convert class IDs to a colored mask.
6. Overlay the mask on the original image.
7. Run MiDaS or another monocular depth model.
8. Normalize and display the depth map.
9. Fuse segmentation and depth for a short scene analysis.
10. Save all outputs.

## Code expectations

Keep code presentation-friendly:

- Use clear variable names: `image_rgb`, `segmentation_mask`, `depth_map`, `overlay`.
- Print shape and output paths.
- Save files into `outputs/`.
- Use comments sparingly, only where helpful for presentation.
- Use Vietnamese notes in notebooks when preparing for class presentation.

## Model guidance

Semantic segmentation:

- Prefer pretrained model for demo if training is too heavy.
- Use U-Net as the explainable baseline architecture.
- If evaluating with ground truth, mention pixel accuracy and mean IoU.

Depth estimation:

- MiDaS produces relative depth in most demo settings.
- Do not claim exact distance in meters without calibration or ground truth.
- Resize the depth map back to the input image size before fusion.

## Common bugs

Check these first:

- BGR/RGB image colors are swapped.
- Segmentation mask and image sizes do not match.
- Depth map shape differs from original image shape.
- Normalization creates unreadable depth display.
- Model input tensor uses wrong shape or channel order.
- Output directory does not exist.

## Validation checklist

Before saying the implementation is ready:

- The notebook/script runs from a clean start.
- At least one street image produces all expected outputs.
- Segmentation overlay is visible and not opaque.
- Depth map is non-empty and visually meaningful.
- Fusion notes do not overclaim exact distances.

