---
title: Traffic Scene Understanding
sdk: gradio
app_file: app.py
python_version: "3.10"
pinned: false
---

# Traffic Scene Understanding

Image demo for the Group 9 traffic scene understanding pipeline:

- road semantic segmentation with U-Net ResNet-50,
- MiDaS relative depth estimation,
- Faster R-CNN MobileNet object detection,
- lane/HUD visualization with relative-distance labels,
- four-panel dashboard output.

MiDaS output represents relative depth within an image. It is not calibrated
distance in meters and must not be used as a production driving-safety system.
