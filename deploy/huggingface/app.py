from functools import lru_cache
import os
from pathlib import Path

import cv2
import gradio as gr
from huggingface_hub import hf_hub_download
import numpy as np
import torch

from model import load_road_model


MODEL_REPO_ID = "xuanhai0913/group9-scene-understanding-models"
ROAD_MODEL_FILENAME = "unet_resnet50_road_state_dict.pth"
MIDAS_MODEL_FILENAME = "midas_v2_1_small.tflite"

SEGMENTATION_SIZE = (1280, 384)
SEGMENTATION_THRESHOLD = -2.5
MAX_OUTPUT_SIDE = 1280

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
    return load_road_model(road_checkpoint), MidasDepthEstimator(
        midas_checkpoint
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


def build_outputs(image_rgb, road_mask, disparity):
    road_color = np.zeros_like(image_rgb)
    road_color[road_mask] = (128, 64, 128)
    segmentation_overlay = cv2.addWeighted(
        image_rgb,
        0.72,
        road_color,
        0.28,
        0,
    )

    depth_bgr = cv2.applyColorMap(disparity, cv2.COLORMAP_MAGMA)
    depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)

    depth_on_road = cv2.addWeighted(
        image_rgb,
        0.52,
        depth_rgb,
        0.48,
        0,
    )
    fusion = image_rgb.copy()
    fusion[road_mask] = depth_on_road[road_mask]

    contours, _ = cv2.findContours(
        road_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(fusion, contours, -1, (255, 255, 255), 2)
    return segmentation_overlay, depth_rgb, fusion


def analyze_scene(image):
    image_rgb = normalize_input(image)
    road_model, depth_estimator = load_runtime()

    road_mask = predict_road_mask(road_model, image_rgb)
    disparity = depth_estimator.predict(image_rgb)
    segmentation_overlay, depth_map, fusion = build_outputs(
        image_rgb,
        road_mask,
        disparity,
    )

    road_coverage = float(road_mask.mean() * 100.0)
    if road_mask.any():
        road_depth = float(np.median(disparity[road_mask]))
        depth_summary = f"Median relative depth on road: {road_depth:.0f}/255"
    else:
        depth_summary = "No stable road region detected"

    summary = (
        f"Road coverage: {road_coverage:.1f}%\n\n"
        f"{depth_summary}\n\n"
        "Depth values are relative within this image, not distance in meters."
    )
    return (
        image_rgb,
        segmentation_overlay,
        depth_map,
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
            run_button = gr.Button("Analyze scene", variant="primary")
            clear_button = gr.ClearButton(
                value="Clear",
                components=[],
            )

    with gr.Tabs():
        with gr.Tab("Overview"):
            with gr.Row():
                original_output = gr.Image(label="Original", height=360)
                fusion_output = gr.Image(label="Fusion", height=360)
        with gr.Tab("Segmentation"):
            segmentation_output = gr.Image(
                label="Road segmentation overlay",
                height=520,
            )
        with gr.Tab("Relative depth"):
            depth_output = gr.Image(
                label="MiDaS relative depth",
                height=520,
            )

    scene_summary = gr.Textbox(
        label="Scene analysis",
        lines=4,
        interactive=False,
    )

    outputs = [
        original_output,
        segmentation_output,
        depth_output,
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

demo.queue(default_concurrency_limit=1, max_size=8)

if __name__ == "__main__":
    demo.launch()
