---
name: group9-scene-understanding
description: Use when working on Group 9's traffic scene understanding project using semantic segmentation and depth estimation. Covers project scope, out-of-scope boundaries, features, datasets, model choices, AI workflow, and presentation-ready explanations.
---

# Group 9 Scene Understanding

## Project context

The project topic is traffic scene understanding from street images using semantic segmentation and depth estimation.

Core idea:

- Semantic segmentation answers: each pixel belongs to which class?
- Depth estimation answers: which regions are relatively near or far from the camera?
- Fusion answers: how can class labels and relative depth help explain the road scene?

Use Vietnamese with proper accents for project-facing documentation unless the user asks otherwise.

## Scope guardrails

Stay inside this scope:

- Input: street/road scene image.
- Output: segmentation mask, segmentation overlay, depth map, and fusion/scene analysis.
- Dataset references: Cityscapes for segmentation, KITTI for traffic/depth context.
- Models: U-Net or pretrained segmentation model; MiDaS for monocular relative depth.
- Goal: a course project demo and explanation, not a production ADAS system.

Keep out of scope unless explicitly requested:

- Real-time autonomous driving decisions.
- Exact metric distance in meters without calibration/ground truth.
- Full training on large datasets if time/hardware is limited.
- Real-time video tracking, object detection, lane violation logic, or control systems.

## Expected document order

When preparing project docs, keep this order:

1. User's requirement
2. Features
3. Tech solutions
4. Logic + AI
5. Implement
6. Test

## Explanation pattern

When summarizing the project, use this structure:

```text
Input: ảnh đường phố.
Segmentation: phân vùng pixel thành road, car, sky, sidewalk...
Depth: tạo depth map để biết vùng gần/xa tương đối.
Fusion: kết hợp class và depth để phân tích ngữ cảnh giao thông.
```

## AI workflow expectation

For AI usage evidence, show that the team:

- Prepared a knowledge base first.
- Asked AI to check requirements and scope.
- Asked AI to compare datasets and models.
- Asked AI to review logic and edge cases.
- Used AI for checklist/test planning.
- Did not ask AI to complete the whole project blindly.

