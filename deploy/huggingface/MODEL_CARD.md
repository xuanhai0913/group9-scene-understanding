---
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - computer-vision
  - semantic-segmentation
  - depth-estimation
  - traffic-scene-understanding
---

# Group 9 Scene Understanding Models

Inference assets for the Group 9 traffic scene understanding course project.

## Files

- `unet_resnet50_road_state_dict.pth`: road segmentation U-Net with a
  ResNet-50 encoder, trained for binary road segmentation.
- `midas_v2_1_small.tflite`: MiDaS v2.1 Small TFLite model used for
  monocular relative depth estimation.

The U-Net checkpoint is an inference-only state dictionary extracted from the
team training checkpoint. Optimizer state and training history are excluded.

## Intended Use

The models support an educational image demo:

1. Segment the drivable road region.
2. Estimate relative depth from a single image.
3. Fuse road segmentation with relative depth for scene visualization.

MiDaS output is relative within each image. It is not calibrated distance in
meters and must not be used for vehicle control or production safety systems.

## References

- [gasparian/multiclass-semantic-segmentation](https://github.com/gasparian/multiclass-semantic-segmentation)
- [ibaiGorordo/Midasv2_1_small-TFLite-Inference](https://github.com/ibaiGorordo/Midasv2_1_small-TFLite-Inference)
