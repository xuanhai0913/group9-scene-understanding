from functools import lru_cache
import os
from pathlib import Path

import cv2
import gradio as gr
from huggingface_hub import hf_hub_download
import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
import torchvision.transforms.functional as F

from model import load_road_model


MODEL_REPO_ID = "xuanhai0913/group9-scene-understanding-models"
ROAD_MODEL_FILENAME = "unet_resnet50_road_state_dict.pth"
MIDAS_MODEL_FILENAME = "midas_v2_1_small.tflite"

SEGMENTATION_SIZE = (1280, 384)
SEGMENTATION_THRESHOLD = -2.5
MAX_OUTPUT_SIDE = 1280
PANEL_SIZE = (480, 270)

VEHICLE_LABELS = {2, 3, 4, 6, 8}
HUMAN_LABELS = {1}
SIGN_LABELS = {10, 13}

RGB_GREEN = (0, 255, 0)
RGB_RED = (255, 0, 0)
RGB_ORANGE = (255, 165, 0)
RGB_YELLOW = (255, 255, 0)
RGB_WHITE = (255, 255, 255)
RGB_BLACK = (0, 0, 0)
RGB_PURPLE = (128, 64, 128)
RGB_DARK_GREEN = (0, 90, 0)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_model_file(filename):
    local_model_dir = os.getenv("MODEL_ASSET_DIR")
    if local_model_dir:
        local_path = Path(local_model_dir) / filename
        if local_path.is_file():
            return str(local_path)

    return hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=filename,
    )


class MidasDepthEstimator:
    def __init__(self, model_path):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=model_path, num_threads=2)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape = self.input_details[0]["shape"]
        output_shape = self.output_details[0]["shape"]
        self.input_height = int(input_shape[1])
        self.input_width = int(input_shape[2])
        self.output_height = int(output_shape[1])
        self.output_width = int(output_shape[2])

    def predict(self, image_rgb):
        resized = cv2.resize(
            image_rgb,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)
        normalized = (resized / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        input_tensor = normalized[np.newaxis, ...].astype(np.float32)

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            input_tensor,
        )
        self.interpreter.invoke()
        raw_disparity = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        ).reshape(self.output_height, self.output_width)

        minimum = float(raw_disparity.min())
        maximum = float(raw_disparity.max())
        scale = max(maximum - minimum, 1e-6)
        disparity = ((raw_disparity - minimum) / scale * 255.0).astype(
            np.uint8
        )
        return cv2.resize(
            disparity,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )


@lru_cache(maxsize=1)
def load_runtime():
    road_checkpoint = resolve_model_file(ROAD_MODEL_FILENAME)
    midas_checkpoint = resolve_model_file(MIDAS_MODEL_FILENAME)
    detector_weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    detector = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=detector_weights,
    )
    detector.eval()
    return (
        load_road_model(road_checkpoint),
        MidasDepthEstimator(midas_checkpoint),
        detector,
    )


def normalize_input(image):
    if image is None:
        raise gr.Error("Vui lòng chọn một ảnh đường phố.")

    image = np.asarray(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image = image.astype(np.uint8)
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side > MAX_OUTPUT_SIDE:
        ratio = MAX_OUTPUT_SIDE / longest_side
        image = cv2.resize(
            image,
            (int(width * ratio), int(height * ratio)),
            interpolation=cv2.INTER_AREA,
        )
    return image


def predict_road_mask(model, image_rgb):
    resized = cv2.resize(
        image_rgb,
        SEGMENTATION_SIZE,
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    normalized = (resized / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = (
        torch.from_numpy(normalized)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )

    with torch.inference_mode():
        logits = model(tensor)
        mask = (logits[0, 0] > SEGMENTATION_THRESHOLD).cpu().numpy()

    return cv2.resize(
        mask.astype(np.uint8),
        (image_rgb.shape[1], image_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def compute_overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0, 0.0

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    min_area = min(area1, area2)
    return (
        intersection / union if union > 0 else 0.0,
        intersection / min_area if min_area > 0 else 0.0,
    )


def obstacle_type(label):
    if label in VEHICLE_LABELS:
        return "vehicle"
    if label in HUMAN_LABELS:
        return "human"
    if label == 10:
        return "traffic light"
    if label == 13:
        return "stop sign"
    return None


def relative_distance(disparity_value):
    return float(1000.0 / (float(disparity_value) + 1e-5))


def detect_obstacles(detector, image_rgb, disparity):
    tensor = F.to_tensor(image_rgb)
    with torch.inference_mode():
        prediction = detector([tensor])[0]

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    obstacles = []
    height, width = image_rgb.shape[:2]
    for box, label, score in zip(boxes, labels, scores):
        obs_type = obstacle_type(int(label))
        if obs_type is None:
            continue

        threshold = 0.30 if int(label) in SIGN_LABELS else 0.20
        if float(score) < threshold:
            continue

        xmin, ymin, xmax, ymax = map(int, np.round(box))
        xmin = int(np.clip(xmin, 0, width - 1))
        xmax = int(np.clip(xmax, 0, width - 1))
        ymin = int(np.clip(ymin, 0, height - 1))
        ymax = int(np.clip(ymax, 0, height - 1))
        if xmax <= xmin or ymax <= ymin:
            continue

        if obs_type == "vehicle" and ymin > int(height * 0.90):
            continue

        overlap = False
        for existing in obstacles:
            if existing["type"] != obs_type:
                continue
            iou, iomin = compute_overlap(
                (xmin, ymin, xmax, ymax),
                existing["box"],
            )
            if obs_type in {"traffic light", "stop sign"}:
                overlap = iou > 0.10 or iomin > 0.35
            else:
                overlap = iou > 0.20 or iomin > 0.45
            if overlap:
                break
        if overlap:
            continue

        depth_crop = disparity[ymin:ymax, xmin:xmax]
        if depth_crop.size == 0:
            continue

        depth_value = float(np.percentile(depth_crop, 95))
        obstacles.append(
            {
                "box": (xmin, ymin, xmax, ymax),
                "depth": depth_value,
                "distance": relative_distance(depth_value),
                "type": obs_type,
                "label": int(label),
                "score": float(score),
            }
        )

    return obstacles


def add_panel_title(image, title):
    panel = image.copy()
    scale = max(0.42, panel.shape[1] / 900.0)
    thickness = max(1, int(panel.shape[1] / 480))
    (text_w, text_h), _ = cv2.getTextSize(
        title,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    x = max(8, (panel.shape[1] - text_w) // 2)
    y = 8 + text_h
    cv2.rectangle(
        panel,
        (x - 8, y - text_h - 6),
        (x + text_w + 8, y + 6),
        RGB_WHITE,
        -1,
    )
    cv2.putText(
        panel,
        title,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        RGB_BLACK,
        thickness,
        cv2.LINE_AA,
    )
    return panel


def make_color_mask(image_rgb, road_mask):
    color_mask = np.zeros_like(image_rgb)
    color_mask[road_mask] = RGB_PURPLE
    return color_mask


def lane_geometry(width, height):
    camera_center = (int(width * 0.70), int(height * 0.92))
    y_top = int(height * 0.43)
    y_bottom = height - 1
    left_top = int(width * 0.48)
    left_bottom = int(width * 0.44)
    right_top = int(width * 0.72)
    right_bottom = int(width * 0.96)
    polygon = np.array(
        [
            [left_bottom, y_bottom],
            [left_top, y_top],
            [right_top, y_top],
            [right_bottom, y_bottom],
        ],
        dtype=np.int32,
    )
    return camera_center, polygon, y_top, y_bottom, left_top, left_bottom


def draw_dashed_line(image, start, end, color, thickness=1, dash=18):
    x1, y1 = start
    x2, y2 = end
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    for offset in range(0, length, dash * 2):
        start_ratio = offset / length
        end_ratio = min(offset + dash, length) / length
        sx = int(x1 + (x2 - x1) * start_ratio)
        sy = int(y1 + (y2 - y1) * start_ratio)
        ex = int(x1 + (x2 - x1) * end_ratio)
        ey = int(y1 + (y2 - y1) * end_ratio)
        cv2.line(image, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)


def is_in_lane(obstacle, lane_polygon, road_mask):
    xmin, ymin, xmax, ymax = obstacle["box"]
    width = road_mask.shape[1]
    height = road_mask.shape[0]
    x_center = int((xmin + xmax) / 2)
    y_contact = int(min(ymax, height - 1))
    x_center = int(np.clip(x_center, 0, width - 1))
    y_contact = int(np.clip(y_contact, 0, height - 1))
    in_polygon = (
        cv2.pointPolygonTest(lane_polygon, (x_center, y_contact), False) >= 0
    )
    on_road = bool(road_mask[y_contact, x_center])
    return in_polygon and on_road


def annotate_scene(image_rgb, color_mask, road_mask, disparity, obstacles):
    height, width = image_rgb.shape[:2]
    output = image_rgb.copy()
    camera_center, lane_polygon, y_top, _, left_top, left_bottom = (
        lane_geometry(width, height)
    )

    lane_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(lane_mask, [lane_polygon], 255)
    lane_mask = (lane_mask > 0) & road_mask
    lane_overlay = output.copy()
    lane_overlay[lane_mask] = RGB_DARK_GREEN
    output = cv2.addWeighted(output, 0.78, lane_overlay, 0.22, 0)

    cv2.polylines(output, [lane_polygon], True, RGB_GREEN, 2, cv2.LINE_AA)
    draw_dashed_line(
        output,
        (10, camera_center[1]),
        (width - 10, camera_center[1]),
        RGB_WHITE,
        max(1, width // 640),
    )
    cv2.putText(
        output,
        "EGO-FRONT BOUNDARY",
        (15, max(18, camera_center[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.42, width / 1900),
        RGB_WHITE,
        max(1, width // 800),
        cv2.LINE_AA,
    )
    cv2.line(
        output,
        (left_bottom, height - 1),
        (left_top, y_top),
        RGB_YELLOW,
        2,
        cv2.LINE_AA,
    )

    lane_obstacles = []
    for obstacle in obstacles:
        obstacle["is_in_lane"] = is_in_lane(obstacle, lane_polygon, road_mask)
        if obstacle["type"] in {"vehicle", "human"} and obstacle["is_in_lane"]:
            lane_obstacles.append(obstacle)

    closest = min(lane_obstacles, key=lambda item: item["distance"], default=None)
    is_warning = closest is not None and closest["distance"] < 5.0
    status_color = RGB_RED if is_warning else RGB_GREEN
    if is_warning:
        status_text = (
            f"WARNING: {closest['type'].upper()} front too close. "
            f"Front rel: {closest['distance']:.1f}"
        )
    elif closest is not None:
        status_text = f"Status: Safe. Front rel: {closest['distance']:.1f}"
    else:
        status_text = "Status: Safe. No obstacles in lane."

    hud_height = max(34, int(height * 0.08))
    hud_overlay = output.copy()
    cv2.rectangle(hud_overlay, (0, 0), (width, hud_height), status_color, -1)
    output = cv2.addWeighted(output, 0.72, hud_overlay, 0.28, 0)
    cv2.putText(
        output,
        status_text,
        (18, int(hud_height * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.48, width / 1500),
        RGB_WHITE,
        max(1, width // 620),
        cv2.LINE_AA,
    )

    for index, obstacle in enumerate(obstacles, start=1):
        xmin, ymin, xmax, ymax = obstacle["box"]
        center = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
        is_closest = closest is obstacle
        if obstacle["type"] in {"traffic light", "stop sign"}:
            box_color = RGB_ORANGE
        elif is_closest and is_warning:
            box_color = RGB_RED
        else:
            box_color = RGB_GREEN

        thickness = max(2, width // 520)
        if is_closest and is_warning:
            thickness += 1

        cv2.rectangle(output, (xmin, ymin), (xmax, ymax), box_color, thickness)
        label = (
            f"{obstacle['type'].upper()} #{index}: "
            f"{obstacle['distance']:.1f} rel"
        )
        font_scale = max(0.42, width / 1800)
        font_thickness = max(1, width // 900)
        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )
        x_label = xmin
        if x_label + text_w + 10 > width - 1:
            x_label = max(0, width - text_w - 12)
        y_label = ymin - 5
        if y_label - text_h - 8 < 0:
            y_label = ymin + text_h + 10
        cv2.rectangle(
            output,
            (x_label, y_label - text_h - 8),
            (min(width - 1, x_label + text_w + 10), y_label + 5),
            RGB_BLACK,
            -1,
        )
        cv2.rectangle(
            output,
            (x_label, y_label - text_h - 8),
            (min(width - 1, x_label + text_w + 10), y_label + 5),
            box_color,
            1,
        )
        cv2.putText(
            output,
            label,
            (x_label + 5, y_label),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            RGB_WHITE,
            font_thickness,
            cv2.LINE_AA,
        )
        if obstacle["type"] in {"vehicle", "human"} and obstacle["is_in_lane"]:
            cv2.line(output, camera_center, center, box_color, 2, cv2.LINE_AA)
            cv2.circle(output, center, 4, box_color, -1)

    car_color = RGB_RED if is_warning else RGB_GREEN
    cv2.circle(output, camera_center, max(9, width // 70), car_color, -1)
    cv2.circle(output, camera_center, max(4, width // 170), RGB_WHITE, 1)
    cv2.putText(
        output,
        "MY CAR",
        (camera_center[0] - 35, camera_center[1] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.44, width / 1800),
        car_color,
        max(1, width // 700),
        cv2.LINE_AA,
    )

    fusion = cv2.addWeighted(output, 0.82, color_mask, 0.18, 0)
    return output, fusion, status_text, lane_obstacles


def build_dashboard(image_rgb, color_mask, depth_rgb, fusion):
    panels = [
        ("Input Image Frame", image_rgb),
        ("Semantic Segmentation (U-Net)", color_mask),
        ("Depth Estimation (MiDaS)", depth_rgb),
        ("Fused Scene Understanding Overlay", fusion),
    ]
    resized = []
    for title, panel in panels:
        view = cv2.resize(panel, PANEL_SIZE, interpolation=cv2.INTER_AREA)
        resized.append(add_panel_title(view, title))
    return np.hstack(resized)


def analyze_scene(image):
    image_rgb = normalize_input(image)
    road_model, depth_estimator, detector = load_runtime()

    road_mask = predict_road_mask(road_model, image_rgb)
    disparity = depth_estimator.predict(image_rgb)
    color_mask = make_color_mask(image_rgb, road_mask)
    depth_bgr = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
    obstacles = detect_obstacles(detector, image_rgb, disparity)
    hud_overlay, fusion, status_text, lane_obstacles = annotate_scene(
        image_rgb,
        color_mask,
        road_mask,
        disparity,
        obstacles,
    )
    dashboard = build_dashboard(image_rgb, color_mask, depth_rgb, fusion)

    road_coverage = float(road_mask.mean() * 100.0)
    nearest = min(
        (obs for obs in obstacles if obs["type"] in {"vehicle", "human"}),
        key=lambda item: item["distance"],
        default=None,
    )
    nearest_text = (
        f"Nearest object: {nearest['type']} at {nearest['distance']:.1f} rel"
        if nearest is not None
        else "Nearest object: none"
    )
    summary = (
        f"{status_text}\n\n"
        f"Detected objects: {len(obstacles)} "
        f"({len(lane_obstacles)} in ego lane)\n"
        f"Road coverage: {road_coverage:.1f}%\n"
        f"{nearest_text}\n\n"
        "Relative distance is a heuristic from MiDaS disparity, not meters."
    )
    return (
        dashboard,
        image_rgb,
        color_mask,
        depth_rgb,
        hud_overlay,
        fusion,
        summary,
    )


with gr.Blocks(title="Traffic Scene Understanding") as demo:
    gr.Markdown("# Traffic Scene Understanding")

    with gr.Row():
        input_image = gr.Image(
            label="Street image",
            type="numpy",
            sources=["upload", "webcam"],
            height=420,
        )
        with gr.Column():
            run_button = gr.Button("Analyze full pipeline", variant="primary")
            clear_button = gr.ClearButton(
                value="Clear",
                components=[],
            )

    dashboard_output = gr.Image(
        label="Full traffic-scene dashboard",
        height=360,
    )

    with gr.Tabs():
        with gr.Tab("Pipeline"):
            with gr.Row():
                original_output = gr.Image(label="Input frame", height=300)
                segmentation_output = gr.Image(
                    label="Semantic segmentation",
                    height=300,
                )
            with gr.Row():
                depth_output = gr.Image(
                    label="MiDaS relative depth",
                    height=300,
                )
                fusion_output = gr.Image(
                    label="Fused scene overlay",
                    height=300,
                )
        with gr.Tab("HUD"):
            hud_output = gr.Image(
                label="Object detection + lane HUD",
                height=520,
            )

    scene_summary = gr.Textbox(
        label="Scene analysis",
        lines=6,
        interactive=False,
    )

    outputs = [
        dashboard_output,
        original_output,
        segmentation_output,
        depth_output,
        hud_output,
        fusion_output,
        scene_summary,
    ]
    run_button.click(
        fn=analyze_scene,
        inputs=input_image,
        outputs=outputs,
        show_progress="full",
    )
    clear_button.add(outputs + [input_image])

demo.queue(default_concurrency_limit=1, max_size=4)

if __name__ == "__main__":
    demo.launch()
