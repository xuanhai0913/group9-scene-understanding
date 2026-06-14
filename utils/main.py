import sys
import os

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
# Insert project root directory and utils directory at the beginning of sys.path to resolve package naming conflicts
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_path = os.path.join(project_root, 'utils')

for path in [project_root, utils_path]:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import ctypes
import threading
import queue
import time

try:
    # When running as a package: python -m utils.main
    from .lane import detect_lanes
    from .tracker import Tracker
    from .model import UnetResNet as Unet
    from .utils import load_train_config
    from .video_loader import get_video_path_interactive, VideoReaderThread
    from .fusion import filter_detections, fuse_detections_and_segmentation
    from .visualization import draw_dashed_rectangle, draw_dashed_line
except ImportError:
    # When running directly: python utils/main.py
    from lane import detect_lanes
    from tracker import Tracker
    from model import UnetResNet as Unet
    from utils import load_train_config
    from video_loader import get_video_path_interactive, VideoReaderThread
    from fusion import filter_detections, fuse_detections_and_segmentation
    from visualization import draw_dashed_rectangle, draw_dashed_line

try:
    # Set DPI awareness for Windows to prevent incorrect window scaling
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

from MidasDepthEstimation.midasDepthEstimator import midasDepthEstimator as MidasDepthEstimator

# 3. DOC VIDEO GIAO THONG DAU VAO (OPENCV) - Dua parser len dau de cau hinh mo hinh
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, default="", help="Duong dan den file video")
parser.add_argument('--high_acc', action='store_true', help="Su dung mo hinh nhan dien SSD300 VGG16 do chinh xac cao")
parser.add_argument('--full_road', action='store_true', help="Giam sat va canh bao va cham tren toan bo long duong (khong chia lan)")
parser.add_argument('--skip_frames', type=int, default=1, help="Chi xu ly moi khung hinh thu N de tang toc tren CPU (skip_frames >= 1)")
args = parser.parse_args()

video_path = args.video_path

# Cau hinh phan cung
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"[INFO] He thong Perception Pipeline dang khoi chay tren: {device}")

# Dinh nghia mau sac hien thi mask cho 8 categories cua Cityscapes (dung dinh dang BGR cho OpenCV)
CLASS_COLORS = [
    (0, 0, 0),          # 0: Void (Den)
    (128, 64, 128),     # 1: Flat/Mat duong (Tim)
    (70, 70, 70),       # 2: Construction/Cong trinh (Xam)
    (153, 153, 153),    # 3: Object/Cot moc/Biển báo (Xam sang) - BGR
    (35, 142, 107),     # 4: Nature/Cay coi (Xanh la) - BGR
    (180, 130, 70),     # 5: Sky/Bau troi (Xanh lam) - BGR
    (60, 20, 220),      # 6: Human/Nguoi di bo (Hong do) - BGR
    (0, 0, 255)         # 7: Vehicle/Xe co (Do) - BGR
]

# 1. KHOI TAO MO HINH SEMANTIC SEGMENTATION (U-Net)
# Chuyen sang dung weights thuc cho "road" (toan bo mat duong) thay vi "lane" (chi rieng 1 lan xe)
unet_weights_path = "weights/UNET_resnet18_road/best_model.pth"
use_fallback_detection = False
detection_model = None

# Doc ten backbone va kich thuoc resize tu file train_config.yaml de khoi tao va chay cho khop
config_path = "config/train_config.yaml"
backbone = "resnext50" # Default
resize_dim = (640, 192) # Default width, height (from [192, 640])
if os.path.exists(config_path):
    try:
        config = load_train_config(config_path)
        backbone = config.get("MODEL", {}).get("backbone", "resnext50")
        cfg_resize = config.get("DATASET", {}).get("resize", [])
        if cfg_resize and len(cfg_resize) == 2:
            # config uses [height, width], cv2.resize uses (width, height)
            resize_dim = (cfg_resize[1], cfg_resize[0])
    except:
        pass

if os.path.exists(unet_weights_path):
    # Load state dict first to inspect number of classes
    state = torch.load(unet_weights_path, map_location=device, weights_only=False)
    state_dict = state.get("state_dict", state)
    
    num_classes = 8  # Default
    if "final.weight" in state_dict:
        num_classes = state_dict["final.weight"].shape[0]
        
    backbone_candidates = [backbone, "resnext50", "resnet18", "resnet34", "resnet50"]
    loaded = False
    for candidate in backbone_candidates:
        try:
            print(f"[INFO] Thu nap checkpoint bang backbone: {candidate}...")
            unet_model = Unet(num_classes=num_classes, encoder_name=candidate).to(device)
            unet_model.load_state_dict(state_dict)
            backbone = candidate
            loaded = True
            print(f"[INFO] Da nap file trong so {unet_weights_path} thanh cong voi backbone: {candidate}")
            break
        except Exception as e:
            print(f"[WARNING] Khong the nap voi backbone {candidate}")
            continue

    if not loaded:
        print(f"[ERROR] Khong the nap file trong so {unet_weights_path} voi bat ky backbone nao!")
        exit(1)
        
    unet_model.eval()
    
    if num_classes == 1:
        print("[INFO] Day la mo hinh phan doan duong nhi phan (1 lop). Nap mo hinh object detection de phat hien phuong tien...")
        if args.high_acc:
            try:
                from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
                detection_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
                print("[INFO] Da nap Faster R-CNN ResNet50 FPN do chinh xac cao.")
            except Exception as e:
                try:
                    from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
                    detection_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT).to(device)
                except Exception as e2:
                    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
                    detection_model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT).to(device)
        else:
            try:
                from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
                detection_model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT).to(device)
            except:
                from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
                detection_model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(device)
        detection_model.eval()
else:
    unet_model = Unet(num_classes=8, encoder_name=backbone).to(device)
    print("[WARNING] Chua co file trong so unet_best.pth trong thu muc 'weights'.")
    print("[WARNING] He thong se tu dong kich hoat Che do Mo phong Thong minh (Simulated Demo Mode) de minh hoa BTL.")
    use_fallback_detection = True
    
    if args.high_acc:
        print("[INFO] Dang nap mo hinh nhan dien do chinh xac cao Faster R-CNN ResNet50 FPN (Co the chay cham hon tren CPU)...")
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
            detection_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
        except Exception as e:
            try:
                from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
                detection_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT).to(device)
            except Exception as e2:
                print(f"[ERROR] Loi khi nap model high_acc: {e2}. Quay lai Faster RCNN MobileNet.")
                from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
                detection_model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT).to(device)
    else:
        print("[INFO] Dang nap mo hinh nhan dien thoi gian thuc Faster RCNN MobileNet...")
        try:
            from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
            detection_model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT).to(device)
        except:
            from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
            detection_model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(device)
    detection_model.eval()

# 2. KHOI TAO MO HINH DEPTH ESTIMATION (MiDaS TFLite cua ibaiGorordo)
try:
    depth_estimator = MidasDepthEstimator()
    print("[INFO] Da nap thanh cong mo hinh MiDaS TFLite.")
except Exception as e:
    print(f"[ERROR] Khong the nap mo hinh MiDaS TFLite. Loi: {e}")
    exit()

if not video_path:
    video_path = get_video_path_interactive(project_root)
    if not video_path:
        # Ultimate fallback
        video_path = "data/sample_videos/video3lightneed.mp4"
        print(f"[INFO] Tự động chọn video mặc định: {video_path}")
else:
    print(f"[INFO] Sử dụng video cấu hình từ đối số: {video_path}")

# if "video3" in video_path.lower() or "lightneed" in video_path.lower():
#     args.full_road = True
#     print("[INFO] Phat hien video3 (duong 2 chieu). Tu dong bat che do không chia lan (full_road = True)!")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"[WARNING] Khong tim thay video hop le tai thu muc data/sample_videos/")
    print("[INFO] He thong tu dong chuyen sang su dung Webcam (Device 0) de kiem thu...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Khong the mo duoc ca video lan Webcam cua may tinh!")
        exit()

# Kiem tra neu chay tren video chuot lang de hien thi ban demo MiDaS chuan
is_midas_demo = "ytsave" in video_path.lower() or "tgadvbd" in video_path.lower()
if is_midas_demo:
    print("[INFO] Phat hien video chuot lang. He thong chuyen sang che do Demo MiDaS (Original | Depth Magma | Blended).")

# Bo tien xu ly chuan hoa anh dau vao cho mang U-Net theo chuan ImageNet
unet_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("[INFO] Dang phan tich luong du lieu... Nhan phim 'q' tai man hinh hien thi de thoat.")

window_name = "BTL Image Processing - Traffic Scene Understanding Pipeline"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

disp_w = None
disp_h = None
scale_x = 1.0
scale_y = 1.0

tracker = Tracker()

# Refactored: compute_overlap, filter_detections, fuse_detections_and_segmentation, and VideoReaderThread moved to utils/fusion.py and utils/video_loader.py


class InferenceThread(threading.Thread):
    def __init__(self, input_queue, output_queue, reader_thread, unet_model, detection_model, depth_estimator, device, resize_dim, use_fallback_detection, unet_transform, is_midas_demo, CLASS_COLORS, args):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.reader_thread = reader_thread
        self.unet_model = unet_model
        self.detection_model = detection_model
        self.depth_estimator = depth_estimator
        self.device = device
        self.resize_dim = resize_dim
        self.use_fallback_detection = use_fallback_detection
        self.unet_transform = unet_transform
        self.is_midas_demo = is_midas_demo
        self.CLASS_COLORS = CLASS_COLORS
        self.args = args
        self.stopped = False
        self.daemon = True

    def run(self):
        # Thiết lập số luồng CPU giới hạn cho PyTorch để giải phóng tài nguyên cho luồng GUI hiển thị
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        while not self.stopped:
            try:
                frame = self.input_queue.get(timeout=0.05)
            except queue.Empty:
                if self.reader_thread.stopped and self.input_queue.empty():
                    self.stopped = True
                    break
                continue

            try:
                orig_h, orig_w = frame.shape[:2]
                
                # Run dynamic lane line detection on the original frame
                pt_left_bottom, pt_left_top, pt_right_bottom, pt_right_top = detect_lanes(frame)

                # --- XU LY DO SAU (MIDAS TFLITE) ---
                depth_map = self.depth_estimator.estimateDepthMap(frame)
                
                if self.is_midas_demo:
                    depth_colored_full = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
                    combinedImg = cv2.addWeighted(frame, 0.7, depth_colored_full, 0.6, 0)
                    
                    self.output_queue.put({
                        'frame': frame,
                        'depth_map': depth_map,
                        'depth_colored_full': depth_colored_full,
                        'combinedImg': combinedImg,
                        'pt_left_bottom': pt_left_bottom,
                        'pt_left_top': pt_left_top,
                        'pt_right_bottom': pt_right_bottom,
                        'pt_right_top': pt_right_top,
                        'detected_obstacles': [],
                        'seg_mask_full': None,
                        'color_mask': None,
                        'is_midas_demo': True
                    })
                    continue

                # Che do Perception Pipeline Giao thong ket hop U-Net phan doan
                if self.use_fallback_detection:
                    seg_mask_full = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    color_mask = np.zeros_like(frame)
                    
                    sky_h = int(orig_h * 0.40)
                    cv2.rectangle(color_mask, (0, 0), (orig_w, sky_h), (180, 130, 70), -1)
                    seg_mask_full[0:sky_h, :] = 5
                    
                    nature_left = np.array([[0, sky_h], [int(orig_w * 0.25), sky_h], [int(orig_w * 0.15), int(orig_h * 0.85)], [0, int(orig_h * 0.85)]], np.int32)
                    nature_right = np.array([[orig_w, sky_h], [int(orig_w * 0.75), sky_h], [int(orig_w * 0.85), int(orig_h * 0.85)], [orig_w, int(orig_h * 0.85)]], np.int32)
                    cv2.fillConvexPoly(color_mask, nature_left, (35, 142, 107))
                    cv2.fillConvexPoly(color_mask, nature_right, (35, 142, 107))
                    seg_mask_full[cv2.drawContours(np.zeros_like(seg_mask_full), [nature_left, nature_right], -1, 1, -1) == 1] = 4

                    road_pts = np.array([[int(orig_w * 0.42), int(orig_h * 0.55)], [int(orig_w * 0.58), int(orig_h * 0.55)], [int(orig_w * 0.90), int(orig_h * 0.95)], [int(orig_w * 0.10), int(orig_h * 0.95)]], np.int32)
                    road_opposite_pts = np.array([[int(orig_w * 0.18), int(orig_h * 0.55)], [int(orig_w * 0.38), int(orig_h * 0.55)], [int(orig_w * 0.45), int(orig_h * 0.95)], [int(orig_w * 0.02), int(orig_h * 0.95)]], np.int32)
                    cv2.fillConvexPoly(color_mask, road_pts, (128, 64, 128))
                    cv2.fillConvexPoly(color_mask, road_opposite_pts, (128, 64, 128))
                    seg_mask_full[cv2.drawContours(np.zeros_like(seg_mask_full), [road_pts, road_opposite_pts], -1, 1, -1) == 1] = 1

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    det_w, det_h = 640, 360
                    frame_det = cv2.resize(frame_rgb, (det_w, det_h))
                    det_tensor = transforms.ToTensor()(frame_det).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        predictions = self.detection_model(det_tensor)[0]
                    
                    boxes = predictions['boxes'].cpu().numpy()
                    labels = predictions['labels'].cpu().numpy()
                    scores = predictions['scores'].cpu().numpy()
                    
                    scale_x_det = orig_w / det_w
                    scale_y_det = orig_h / det_h
                    
                    detected_obstacles = filter_detections(boxes, labels, scores, orig_w, orig_h, scale_x_det, scale_y_det, depth_map)
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    seg_resized = cv2.resize(frame_rgb, self.resize_dim) 
                    seg_tensor = self.unet_transform(seg_resized).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        seg_output = self.unet_model(seg_tensor)
                        if seg_output.shape[1] == 1:
                            seg_mask = (torch.sigmoid(seg_output) > 0.5).long().squeeze(0).squeeze(0).cpu().numpy()
                        else:
                            seg_mask = torch.argmax(seg_output, dim=1).squeeze(0).cpu().numpy()

                    seg_mask_unet = cv2.resize(seg_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    seg_mask_full = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    
                    sky_h = int(orig_h * 0.40)
                    seg_mask_full[0:sky_h, :] = 5
                    
                    nature_left = np.array([[0, sky_h], [int(orig_w * 0.25), sky_h], [int(orig_w * 0.15), int(orig_h * 0.85)], [0, int(orig_h * 0.85)]], np.int32)
                    nature_right = np.array([[orig_w, sky_h], [int(orig_w * 0.75), sky_h], [int(orig_w * 0.85), int(orig_h * 0.85)], [orig_w, int(orig_h * 0.85)]], np.int32)
                    nature_mask = np.zeros_like(seg_mask_full)
                    cv2.drawContours(nature_mask, [nature_left, nature_right], -1, 1, -1)
                    seg_mask_full[nature_mask == 1] = 4
                    
                    if seg_output.shape[1] == 1:
                        seg_mask_full[seg_mask_unet == 1] = 1
                    else:
                        seg_mask_full = seg_mask_unet

                    detected_obstacles = []
                    if self.detection_model is not None:
                        det_w, det_h = 640, 360
                        frame_det = cv2.resize(frame_rgb, (det_w, det_h))
                        det_tensor = transforms.ToTensor()(frame_det).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            predictions = self.detection_model(det_tensor)[0]
                        
                        boxes = predictions['boxes'].cpu().numpy()
                        labels = predictions['labels'].cpu().numpy()
                        scores = predictions['scores'].cpu().numpy()
                        
                        scale_x_det = orig_w / det_w
                        scale_y_det = orig_h / det_h
                        
                        detected_obstacles = filter_detections(boxes, labels, scores, orig_w, orig_h, scale_x_det, scale_y_det, depth_map)

                # Run Fusion
                fused_obstacles = fuse_detections_and_segmentation(
                    detected_obstacles, seg_mask_full, depth_map,
                    self.use_fallback_detection, num_classes
                )

                # Apply fused targets back onto seg_mask_full & color_mask
                if self.use_fallback_detection:
                    color_mask = np.zeros_like(frame)
                    sky_h = int(orig_h * 0.40)
                    cv2.rectangle(color_mask, (0, 0), (orig_w, sky_h), (180, 130, 70), -1)
                    nature_left = np.array([[0, sky_h], [int(orig_w * 0.25), sky_h], [int(orig_w * 0.15), int(orig_h * 0.85)], [0, int(orig_h * 0.85)]], np.int32)
                    nature_right = np.array([[orig_w, sky_h], [int(orig_w * 0.75), sky_h], [int(orig_w * 0.85), int(orig_h * 0.85)], [orig_w, int(orig_h * 0.85)]], np.int32)
                    cv2.fillConvexPoly(color_mask, nature_left, (35, 142, 107))
                    cv2.fillConvexPoly(color_mask, nature_right, (35, 142, 107))
                    
                    road_pts = np.array([[int(orig_w * 0.42), int(orig_h * 0.55)], [int(orig_w * 0.58), int(orig_h * 0.55)], [int(orig_w * 0.90), int(orig_h * 0.95)], [int(orig_w * 0.10), int(orig_h * 0.95)]], np.int32)
                    road_opposite_pts = np.array([[int(orig_w * 0.18), int(orig_h * 0.55)], [int(orig_w * 0.38), int(orig_h * 0.55)], [int(orig_w * 0.45), int(orig_h * 0.95)], [int(orig_w * 0.02), int(orig_h * 0.95)]], np.int32)
                    cv2.fillConvexPoly(color_mask, road_pts, (128, 64, 128))
                    cv2.fillConvexPoly(color_mask, road_opposite_pts, (128, 64, 128))
                    
                    for det in fused_obstacles:
                        xmin, ymin, xmax, ymax = det['box']
                        obs_type = det['type']
                        if obs_type == 'vehicle':
                            cv2.rectangle(color_mask, (xmin, ymin), (xmax, ymax), (0, 0, 255), -1)
                            seg_mask_full[ymin:ymax, xmin:xmax] = 7
                        elif obs_type == 'human':
                            cv2.rectangle(color_mask, (xmin, ymin), (xmax, ymax), (60, 20, 220), -1)
                            seg_mask_full[ymin:ymax, xmin:xmax] = 6
                        elif obs_type in ['traffic light', 'stop sign']:
                            cv2.rectangle(color_mask, (xmin, ymin), (xmax, ymax), (153, 153, 153), -1)
                            seg_mask_full[ymin:ymax, xmin:xmax] = 3
                else:
                    for det in fused_obstacles:
                        xmin, ymin, xmax, ymax = det['box']
                        obs_type = det['type']
                        if obs_type == 'vehicle':
                            seg_mask_full[ymin:ymax, xmin:xmax] = 7
                        elif obs_type == 'human':
                            seg_mask_full[ymin:ymax, xmin:xmax] = 6
                        elif obs_type in ['traffic light', 'stop sign']:
                            seg_mask_full[ymin:ymax, xmin:xmax] = 3
                    
                    color_mask = np.zeros_like(frame)
                    for class_idx, color in enumerate(self.CLASS_COLORS):
                        color_mask[seg_mask_full == class_idx] = color

                depth_colored_full = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

                self.output_queue.put({
                    'frame': frame,
                    'depth_map': depth_map,
                    'depth_colored_full': depth_colored_full,
                    'seg_mask_full': seg_mask_full,
                    'color_mask': color_mask,
                    'detected_obstacles': fused_obstacles,
                    'pt_left_bottom': pt_left_bottom,
                    'pt_left_top': pt_left_top,
                    'pt_right_bottom': pt_right_bottom,
                    'pt_right_top': pt_right_top,
                    'is_midas_demo': False
                })
            except Exception as thread_err:
                import traceback
                print("\n[ERROR] EXCEPTION IN INFERENCE THREAD:")
                traceback.print_exc()

    def stop(self):
        self.stopped = True

# Refactored: draw_dashed_rectangle and draw_dashed_line moved to utils/visualization.py


# Khoi tao queues
input_queue = queue.Queue(maxsize=3)
result_queue = queue.Queue(maxsize=3)

# Khoi dong VideoReaderThread
reader_thread = VideoReaderThread(cap, queue_maxsize=3, skip_frames=args.skip_frames)
reader_thread.start()

# Khoi dong InferenceThread
inference_thread = InferenceThread(
    input_queue=reader_thread.queue,
    output_queue=result_queue,
    reader_thread=reader_thread,
    unet_model=unet_model,
    detection_model=detection_model,
    depth_estimator=depth_estimator,
    device=device,
    resize_dim=resize_dim,
    use_fallback_detection=use_fallback_detection,
    unet_transform=unet_transform,
    is_midas_demo=is_midas_demo,
    CLASS_COLORS=CLASS_COLORS,
    args=args
)
inference_thread.start()

try:
    while True:
        try:
            # Lay ket qua suy luan tu result_queue
            result = result_queue.get(timeout=0.02)
        except queue.Empty:
            if reader_thread.stopped and reader_thread.queue.empty() and result_queue.empty():
                break
            # Neu cua so OpenCV dang mo, van can bat ky de tranh cua so bi treo (waitKey)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        frame = result['frame']
        depth_map = result['depth_map']
        depth_colored_full = result['depth_colored_full']
        pt_left_bottom = result['pt_left_bottom']
        pt_left_top = result['pt_left_top']
        pt_right_bottom = result['pt_right_bottom']
        pt_right_top = result['pt_right_top']
        is_midas_demo = result['is_midas_demo']
        
        orig_h, orig_w = frame.shape[:2]

        if disp_w is None:
            try:
                user32 = ctypes.windll.user32
                screen_w = user32.GetSystemMetrics(0)
                screen_h = user32.GetSystemMetrics(1)
            except Exception:
                screen_w = 1280
                screen_h = 720
            
            # Max total width is 75% of screen width to fit nicely
            max_total_w = int(screen_w * 0.75)
            max_h = int(screen_h * 0.6)
            
            # Calculate width for a single panel
            target_w = max_total_w // 3
            target_h = int(target_w * orig_h / orig_w)
            
            if target_h > max_h:
                target_h = max_h
                target_w = int(target_h * orig_w / orig_h)
            
            disp_w, disp_h = target_w, target_h
            cv2.resizeWindow(window_name, disp_w * 3, disp_h)

        # Fetch actual window client area dimensions dynamically to adapt to resizing
        try:
            rect = cv2.getWindowImageRect(window_name)
            if rect is not None and rect[2] > 100 and rect[3] > 100:
                win_w, win_h = rect[2], rect[3]
            else:
                win_w, win_h = disp_w * 3, disp_h
        except Exception:
            win_w, win_h = disp_w * 3, disp_h

        # Preserve the aspect ratio of the 3 panels combined inside the window client area
        dash_aspect = 3.0 * (orig_w / orig_h)
        win_aspect = win_w / win_h
        
        if win_aspect > dash_aspect:
            disp_h = win_h
            disp_w = int(disp_h * orig_w / orig_h)
        else:
            disp_w = win_w // 3
            disp_h = int(disp_w * orig_h / orig_w)
            
        disp_w = max(160, disp_w)
        disp_h = max(120, disp_h)
        
        scale_x = disp_w / orig_w
        scale_y = disp_h / orig_h

        if is_midas_demo:
            combinedImg = result['combinedImg']
            view_main = cv2.resize(frame, (disp_w, disp_h))
            view_depth = cv2.resize(depth_colored_full, (disp_w, disp_h))
            view_combined = cv2.resize(combinedImg, (disp_w, disp_h))
            dashboard = np.hstack((view_main, view_depth, view_combined))
        else:
            seg_mask_full = result['seg_mask_full']
            color_mask = result['color_mask']
            detected_obstacles = result['detected_obstacles']
            
            view_main = cv2.resize(frame, (disp_w, disp_h))
            view_seg = cv2.resize(color_mask, (disp_w, disp_h))
            view_depth = cv2.resize(depth_colored_full, (disp_w, disp_h))
            
            output_frame = cv2.addWeighted(view_main, 0.7, view_seg, 0.3, 0)
            y_top = int(disp_h * 0.45)
            y_bottom = disp_h
            x_top = int(disp_w * 0.50)
            x_bottom = int(disp_w * 0.45)

            # UPDATE TRACKER AND RUN DYNAMIC EGO-CORRIDOR THREAT CHECK
            active_tracks = tracker.update(detected_obstacles)
            
            # Tự động nhận diện đường 2 chiều (oncoming traffic on the left / yellow lane divider color check)
            if not args.full_road and not getattr(tracker, 'auto_full_road_detected', False):
                if not hasattr(tracker, 'oncoming_hits_count'):
                    tracker.oncoming_hits_count = 0
                    tracker.processed_frames_count = 0
                    tracker.yellow_line_check_count = 0
                
                tracker.processed_frames_count += 1
                
                # 1. Kiểm tra màu vạch kẻ làn ở giữa trong 30 khung hình đầu tiên bằng cách đếm số điểm màu vàng trong vùng trung tâm
                if tracker.processed_frames_count <= 30:
                    try:
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        # Vùng trung tâm mặt đường nghi ngờ có vạch phân chia làn
                        y1_crop = int(orig_h * 0.55)
                        y2_crop = int(orig_h * 0.95)
                        x1_crop = int(orig_w * 0.35)
                        x2_crop = int(orig_w * 0.52)
                        
                        hsv_region = hsv[y1_crop:y2_crop, x1_crop:x2_crop]
                        if hsv_region.size > 0:
                            mask = cv2.inRange(hsv_region, np.array([5, 40, 80]), np.array([38, 255, 255]))
                            yellow_pixels = np.sum(mask > 0)
                            if yellow_pixels >= 100:
                                tracker.yellow_line_check_count += 1

                    except Exception:
                        pass
                    
                    if tracker.yellow_line_check_count >= 5:
                        tracker.auto_full_road_detected = True
                        print("[AUTO-DETECTION] Phat hien vach ke duong mau vang (vach phan chia 2 chieu). Tu dong bat che do duong 2 chieu (full_road = True)!")
                
                # 2. Thuật toán động theo dõi xe ngược chiều ở làn trái (dự phòng)
                for tid, track in active_tracks.items():
                    if track['type'] == 'vehicle':
                        xmin_orig, ymin_orig, xmax_orig, ymax_orig = track['box']
                        x_center_orig = (xmin_orig + xmax_orig) // 2
                        # Xe ngược chiều đi bên trái (x_center_orig < orig_w * 0.46) và ở xa/vừa phải (ymin_orig > orig_h * 0.35)
                        if x_center_orig < orig_w * 0.46 and ymin_orig > orig_h * 0.35:
                            depth_hist = track['depth_history']
                            if len(depth_hist) >= 2:
                                dist_hist = [1000.0 / (d + 1e-5) for d in depth_hist]
                                diff = dist_hist[-1] - dist_hist[-2]
                                # Tốc độ tiếp cận nhanh (khoảng cách giảm > 3m trong 1 chu kỳ)
                                if diff < -3.0:
                                    tracker.oncoming_hits_count += 1
                                    
                # Quyết định chế độ đường sau 45 khung hình đầu tiên nếu chưa nhận diện được bằng vạch kẻ đường
                if tracker.processed_frames_count == 45:
                    if tracker.oncoming_hits_count >= 8:
                        tracker.auto_full_road_detected = True
                        print("[AUTO-DETECTION] Phat hien xe nguoc chieu tren lan trai. Tu dong bat che do duong 2 chieu (full_road = True)!")
                    else:
                        print("[AUTO-DETECTION] Khong phat hien xe nguoc chieu. Duy tri che do duong 1 chieu (full_road = False).")
                                        
            is_full_road = args.full_road or getattr(tracker, 'auto_full_road_detected', False)
            
            if is_full_road:
                camera_center = (int(disp_w * 0.50), int(disp_h * 0.92))
            else:
                camera_center = (int(disp_w * 0.70), int(disp_h * 0.92))
            
            path_obstacles_above = []
            path_obstacles_below = []
            cutting_in_tracks = []
            
            h, w = seg_mask_full.shape[:2]
            
            if active_tracks:
                for tid, track in active_tracks.items():
                    xmin_orig, ymin_orig, xmax_orig, ymax_orig = track['box']
                    x_center_orig = (xmin_orig + xmax_orig) // 2
                    y_center_orig = (ymin_orig + ymax_orig) // 2
                    
                    if is_full_road:
                        ego_poly_orig = np.array([
                            (int(w * 0.15), int(h * 0.95)),
                            (int(w * 0.40), int(h * 0.55)),
                            (int(w * 0.65), int(h * 0.55)),
                            (int(w * 0.90), int(h * 0.95))
                        ], dtype=np.int32)
                        is_in_lane = (cv2.pointPolygonTest(ego_poly_orig, (x_center_orig, y_center_orig), False) >= 0)
                    else:
                        if pt_left_bottom and pt_left_top and pt_right_bottom and pt_right_top:
                            x1_l, y1_l = pt_left_bottom
                            x2_l, y2_l = pt_left_top
                            x1_r, y1_r = pt_right_bottom
                            x2_r, y2_r = pt_right_top
                        else:
                            x1_l, y1_l = int(w * 0.46), int(h * 0.95)
                            x2_l, y2_l = int(w * 0.48), int(h * 0.55)
                            x1_r, y1_r = int(w * 0.95), int(h * 0.95)
                            x2_r, y2_r = int(w * 0.62), int(h * 0.55)
                            
                        if abs(y1_l - y2_l) > 0:
                            x_div_left = x2_l + (y_center_orig - y2_l) * (x1_l - x2_l) / (y1_l - y2_l)
                        else:
                            x_div_left = x2_l
                            
                        if abs(y1_r - y2_r) > 0:
                            x_div_right = x2_r + (y_center_orig - y2_r) * (x1_r - x2_r) / (y1_r - y2_r)
                        else:
                            x_div_right = x2_r
                            
                        is_in_lane = (x_center_orig >= x_div_left) and (x_center_orig <= x_div_right)
                    
                    dist_curr = 1000.0 / (track['depth_history'][-1] + 1e-5)
                    delta_d = track.get('delta_d', 0.0)
                    
                    track['dist'] = dist_curr
                    track['is_in_lane'] = is_in_lane
                    
                    if track['type'] in ['vehicle', 'human']:
                        if is_in_lane:
                            orig_camera_center_y = int(h * 0.92)
                            if y_center_orig <= orig_camera_center_y:
                                path_obstacles_above.append(track)
                            else:
                                path_obstacles_below.append(track)
                        else:
                            if dist_curr < 12.0 and delta_d > 0.35:
                                if pt_left_bottom and pt_left_top and pt_right_bottom and pt_right_top:
                                    ego_poly_orig = np.array([pt_left_bottom, pt_left_top, pt_right_top, pt_right_bottom], dtype=np.int32)
                                else:
                                    ego_poly_orig = np.array([
                                        (int(w * 0.38), int(h * 0.95)),
                                        (int(w * 0.46), int(h * 0.55)),
                                        (int(w * 0.62), int(h * 0.55)),
                                        (int(w * 0.92), int(h * 0.95))
                                    ], dtype=np.int32)
                                
                                dist_to_lane = abs(cv2.pointPolygonTest(ego_poly_orig, (x_center_orig, ymax_orig), True))
                                if dist_to_lane < 35:
                                    box_history = track.get('box_history', [])
                                    is_moving_towards_lane = False
                                    if len(box_history) >= 3:
                                        x_prev = (box_history[-3][0] + box_history[-3][2]) // 2
                                        lane_center_x = (ego_poly_orig[1][0] + ego_poly_orig[2][0]) // 2
                                        if x_center_orig < lane_center_x:
                                            is_moving_towards_lane = (x_center_orig > x_prev + 4)
                                        else:
                                            is_moving_towards_lane = (x_center_orig < x_prev - 4)
                                    
                                    if is_moving_towards_lane:
                                        cutting_in_tracks.append((tid, dist_curr))

            closest_above = None
            closest_below = None

            if path_obstacles_above:
                path_obstacles_above.sort(key=lambda x: x['dist'])
                closest_above = path_obstacles_above[0]
                
            if path_obstacles_below:
                path_obstacles_below.sort(key=lambda x: x['dist'])
                closest_below = path_obstacles_below[0]

            dist_above = closest_above['dist'] if closest_above is not None else None
            dist_below = closest_below['dist'] if closest_below is not None else None

            warn_above = dist_above is not None and dist_above < 5.0
            warn_below = dist_below is not None and dist_below < 5.0
            is_warning = warn_above or warn_below

            # 1. Ve duong ranh gioi mui xe va duong phan lan ky thuat so
            annot_scale = max(0.35, 0.45 * (disp_h / 360.0))
            annot_thickness = max(1, int(1.5 * (disp_h / 360.0)))

            for x in range(0, disp_w, 20):
                cv2.line(output_frame, (x, camera_center[1]), (min(x + 10, disp_w), camera_center[1]), (255, 255, 255), 1)
            cv2.putText(output_frame, "EGO-FRONT BOUNDARY", (15, camera_center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, annot_scale, (255, 255, 255), annot_thickness, cv2.LINE_AA)

            disp_pt_left_bottom = (int(pt_left_bottom[0] * scale_x), int(pt_left_bottom[1] * scale_y)) if pt_left_bottom else None
            disp_pt_left_top = (int(pt_left_top[0] * scale_x), int(pt_left_top[1] * scale_y)) if pt_left_top else None
            disp_pt_right_bottom = (int(pt_right_bottom[0] * scale_x), int(pt_right_bottom[1] * scale_y)) if pt_right_bottom else None
            disp_pt_right_top = (int(pt_right_top[0] * scale_x), int(pt_right_top[1] * scale_y)) if pt_right_top else None

            overlay = output_frame.copy()
            if is_full_road:
                disp_ego_poly = np.array([
                    (int(disp_w * 0.15), int(disp_h * 0.95)),
                    (int(disp_w * 0.40), int(disp_h * 0.55)),
                    (int(disp_w * 0.65), int(disp_h * 0.55)),
                    (int(disp_w * 0.90), int(disp_h * 0.95))
                ], dtype=np.int32)
                cv2.fillPoly(overlay, [disp_ego_poly], (0, 255, 0))
                cv2.addWeighted(overlay, 0.15, output_frame, 0.85, 0, output_frame)
                cv2.line(output_frame, disp_ego_poly[0], disp_ego_poly[1], (0, 255, 0), 2, cv2.LINE_AA)
                cv2.line(output_frame, disp_ego_poly[3], disp_ego_poly[2], (0, 255, 0), 2, cv2.LINE_AA)
            else:
                if pt_left_bottom and pt_left_top and pt_right_bottom and pt_right_top:
                    lane_pts_disp = np.array([disp_pt_left_bottom, disp_pt_left_top, disp_pt_right_top, disp_pt_right_bottom], dtype=np.int32)
                    cv2.fillPoly(overlay, [lane_pts_disp], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.15, output_frame, 0.85, 0, output_frame)
                    cv2.line(output_frame, disp_pt_left_bottom, disp_pt_left_top, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.line(output_frame, disp_pt_right_bottom, disp_pt_right_top, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    p1, p2 = disp_pt_left_bottom, disp_pt_left_top
                    num_segments = 15
                    for i in range(num_segments):
                        t1 = i / num_segments
                        t2 = min(1.0, (i + 0.5) / num_segments)
                        sub_pt1 = (int(p1[0] + t1 * (p2[0] - p1[0])), int(p1[1] + t1 * (p2[1] - p1[1])))
                        sub_pt2 = (int(p1[0] + t2 * (p2[0] - p1[0])), int(p1[1] + t2 * (p2[1] - p1[1])))
                        cv2.line(output_frame, sub_pt1, sub_pt2, (0, 255, 255), 2)
                    cv2.putText(output_frame, "LANE SEPARATOR", (disp_pt_left_top[0] - int(10 * (disp_w / 640.0)), disp_pt_left_top[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, annot_scale, (0, 255, 255), annot_thickness, cv2.LINE_AA)
                else:
                    lane_pts_disp = np.array([
                        (int(disp_w * 0.38), int(disp_h * 0.95)),
                        (int(disp_w * 0.46), int(disp_h * 0.55)),
                        (int(disp_w * 0.62), int(disp_h * 0.55)),
                        (int(disp_w * 0.92), int(disp_h * 0.95))
                    ], dtype=np.int32)
                    cv2.fillPoly(overlay, [lane_pts_disp], (128, 64, 128))
                    cv2.addWeighted(overlay, 0.1, output_frame, 0.9, 0, output_frame)
                    
                    for y_draw in range(y_top, y_bottom, 15):
                        r1 = (y_draw - y_top) / (y_bottom - y_top)
                        r2 = (min(y_draw + 8, y_bottom) - y_top) / (y_bottom - y_top)
                        pt1 = (int(x_top + r1 * (x_bottom - x_top)), y_draw)
                        pt2 = (int(x_top + r2 * (x_bottom - x_top)), min(y_draw + 8, y_bottom))
                        cv2.line(output_frame, pt1, pt2, (0, 255, 255), 2)
                    lane_sep_x = max(10, x_top - int(120 * (disp_w / 640.0)))
                    cv2.putText(output_frame, "LANE SEPARATOR", (lane_sep_x, y_top + 20), cv2.FONT_HERSHEY_SIMPLEX, annot_scale, (0, 255, 255), annot_thickness, cv2.LINE_AA)

            # 2. HUD Banner warnings
            hud_h = int(disp_h * 0.15)
            hud_font_scale = max(0.45, 0.6 * (disp_h / 360.0))
            hud_thickness = 2 if hud_font_scale >= 0.55 else 1

            if is_warning:
                cv2.rectangle(output_frame, (0, 0), (disp_w, disp_h), (0, 0, 255), 8)
                overlay = output_frame.copy()
                cv2.rectangle(overlay, (0, 0), (disp_w, hud_h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.4, output_frame, 0.6, 0, output_frame)
                
                msg = "COLLISION WARNING: Obstacle too close!"
                if warn_above and dist_above is not None:
                    msg += f" Front: {dist_above:.1f}m"
                if warn_below and dist_below is not None:
                    msg += f" Below: {dist_below:.1f}m"
                cv2.putText(output_frame, msg, (15, int(hud_h * 0.7)), cv2.FONT_HERSHEY_DUPLEX, hud_font_scale, (255, 255, 255), hud_thickness, cv2.LINE_AA)
            elif cutting_in_tracks:
                cv2.rectangle(output_frame, (0, 0), (disp_w, disp_h), (0, 165, 255), 6)
                overlay = output_frame.copy()
                cv2.rectangle(overlay, (0, 0), (disp_w, hud_h), (0, 165, 255), -1)
                cv2.addWeighted(overlay, 0.4, output_frame, 0.6, 0, output_frame)
                
                cutting_in_tracks.sort(key=lambda x: x[1])
                tid_cut, dist_cut = cutting_in_tracks[0]
                track_type = active_tracks[tid_cut]['type'].upper() if tid_cut in active_tracks else "OBJECT"
                msg = f"WARNING: {track_type} #{tid_cut} is cutting in! Dist: {dist_cut:.1f}m"
                cv2.putText(output_frame, msg, (15, int(hud_h * 0.7)), cv2.FONT_HERSHEY_DUPLEX, hud_font_scale, (255, 255, 255), hud_thickness, cv2.LINE_AA)
            else:
                overlay = output_frame.copy()
                cv2.rectangle(overlay, (0, 0), (disp_w, hud_h), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.2, output_frame, 0.8, 0, output_frame)
                
                msg = "Status: Safe."
                if dist_above is not None:
                    msg += f" Front: {dist_above:.1f}m"
                if dist_below is not None:
                    msg += f" Close: {dist_below:.1f}m"
                if dist_above is None and dist_below is None:
                    msg += " No obstacles in lane."
                cv2.putText(output_frame, msg, (15, int(hud_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, (255, 255, 255), hud_thickness, cv2.LINE_AA)

            # 3. Ve tat ca chuong ngai vat va khoang cach co dung ID tu Tracker
            if active_tracks:
                for tid, track in active_tracks.items():
                    xmin_orig, ymin_orig, xmax_orig, ymax_orig = track['box']
                    xmin = int(xmin_orig * scale_x)
                    ymin = int(ymin_orig * scale_y)
                    xmax = int(xmax_orig * scale_x)
                    ymax = int(ymax_orig * scale_y)
                    x_center = (xmin + xmax) // 2
                    y_center = (ymin + ymax) // 2
                    
                    dist = track['dist']
                    is_in_lane = track['is_in_lane']
                    
                    # Cảnh báo bất kỳ xe nào trong làn có khoảng cách dưới ngưỡng an toàn
                    is_danger = False
                    if is_in_lane:
                        if dist < 5.0:
                            is_danger = True
                        
                    is_cutting_in = any(x[0] == tid for x in cutting_in_tracks)
                    
                    if is_danger:
                        color = (0, 0, 255) # Red
                        thickness = 3
                    elif is_cutting_in:
                        color = (0, 165, 255) # Orange
                        thickness = 2
                    else:
                        color = (0, 255, 0) # Green
                        thickness = 1
                        
                    is_lost = (track.get('age', 0) > 0)
                    if is_lost:
                        draw_dashed_rectangle(output_frame, (xmin, ymin), (xmax, ymax), color, thickness)
                    else:
                        cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), color, thickness)
                    
                    # Mark if the vehicle was recovered from segmentation (fused backup) or estimated via motion tracking
                    if is_lost:
                        suffix = " [EST]"
                    elif track.get('fused_backup', False):
                        suffix = " [F]"
                    else:
                        suffix = ""
                        
                    label_text = f"{track['type'].upper()}{suffix} #{tid}: {dist:.1f}m"
                    font_scale = max(0.35, 0.5 * (disp_h / 360.0))
                    font_thickness = max(1, int(1.5 * (disp_h / 360.0)))
                    (w_label, h_label), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    
                    y_label = ymin - 4
                    if y_label - h_label - 4 < 0:
                        y_label = ymin + h_label + 8
                        
                    cv2.rectangle(output_frame, (xmin, y_label - h_label - 4), (xmin + w_label + 10, y_label + 4), (0, 0, 0), -1)
                    if is_lost:
                        draw_dashed_rectangle(output_frame, (xmin, y_label - h_label - 4), (xmin + w_label + 10, y_label + 4), color, 1)
                    else:
                        cv2.rectangle(output_frame, (xmin, y_label - h_label - 4), (xmin + w_label + 10, y_label + 4), color, 1)
                    cv2.putText(output_frame, label_text, (xmin + 5, y_label), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                    
                    if track['type'] in ['vehicle', 'human'] and (is_full_road or track['is_in_lane'] or is_cutting_in):
                        line_thickness = 2 if (is_danger or is_cutting_in) else 1
                        if is_lost:
                            draw_dashed_line(output_frame, camera_center, (x_center, y_center), color, line_thickness)
                        else:
                            cv2.line(output_frame, camera_center, (x_center, y_center), color, line_thickness)
                        cv2.circle(output_frame, (x_center, y_center), 4, color, -1)

            # 5. Ve camera (MY CAR) dong radar chuyen dong bat mat
            car_color = (0, 0, 255) if is_warning else ((0, 165, 255) if cutting_in_tracks else (0, 255, 0))
            overlay_car = output_frame.copy()
            
            car_radius_outer = max(10, int(20 * (disp_h / 360.0)))
            car_radius_inner = max(4, int(8 * (disp_h / 360.0)))
            
            cv2.circle(overlay_car, camera_center, car_radius_outer, car_color, -1)
            cv2.addWeighted(overlay_car, 0.25, output_frame, 0.75, 0, output_frame)
            cv2.circle(output_frame, camera_center, car_radius_inner, car_color, -1)
            cv2.circle(output_frame, camera_center, car_radius_inner, (255, 255, 255), 1)
            
            my_car_scale = max(0.4, 0.5 * (disp_h / 360.0))
            my_car_thickness = max(1, int(2 * (disp_h / 360.0)))
            cv2.putText(output_frame, "MY CAR", (camera_center[0] - int(30 * (disp_w / 640.0)), camera_center[1] - int(25 * (disp_h / 360.0))), cv2.FONT_HERSHEY_SIMPLEX, my_car_scale, car_color, my_car_thickness, cv2.LINE_AA)

            dashboard = np.hstack((output_frame, view_seg, view_depth))

        # Centering and showing
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        y_offset = max(0, (win_h - disp_h) // 2)
        x_offset = max(0, (win_w - disp_w * 3) // 2)
        
        h_draw = min(disp_h, win_h - y_offset)
        w_draw = min(disp_w * 3, win_w - x_offset)
        
        if h_draw > 0 and w_draw > 0:
            canvas[y_offset:y_offset+h_draw, x_offset:x_offset+w_draw] = dashboard[:h_draw, :w_draw]
            
        cv2.imshow("BTL Image Processing - Traffic Scene Understanding Pipeline", canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            try:
                cv2.imwrite("original_image.png", frame)
                if 'depth_colored_full' in locals() and depth_colored_full is not None:
                    cv2.imwrite("depth_map.png", depth_colored_full)
                if 'color_mask' in locals() and color_mask is not None:
                    seg_overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
                    cv2.imwrite("segmentation_overlay.png", seg_overlay)
                    cv2.imwrite("segmentation_color_mask.png", color_mask)
                if 'output_frame' in locals() and output_frame is not None:
                    cv2.imwrite("fusion_result.png", output_frame)
                if 'dashboard' in locals() and dashboard is not None:
                    cv2.imwrite("dashboard_result.png", dashboard)
                print("\n[INFO] Da chup anh va luu ket qua khung hinh hien tai vao cac file .png thanh cong!")
            except Exception as e:
                print(f"\n[ERROR] Khong the luu anh: {e}")

    # Hoi nguoi dung co muon luu lai khung hinh cuoi cung khi thoat khong
    try:
        if 'frame' in locals() and frame is not None:
            print("\n" + "="*70)
            print("                 LUU ANH KET QUA PHAN TICH KHUNG HINH CUOI CUNG")
            print("="*70)
            choice = input("Ban co muon luu lai anh ket qua cua khung hinh cuoi cung khong? (y/n): ").strip().lower()
            if choice == 'y':
                cv2.imwrite("original_image.png", frame)
                if 'depth_colored_full' in locals() and depth_colored_full is not None:
                    cv2.imwrite("depth_map.png", depth_colored_full)
                if 'color_mask' in locals() and color_mask is not None:
                    seg_overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
                    cv2.imwrite("segmentation_overlay.png", seg_overlay)
                    cv2.imwrite("segmentation_color_mask.png", color_mask)
                if 'output_frame' in locals() and output_frame is not None:
                    cv2.imwrite("fusion_result.png", output_frame)
                if 'dashboard' in locals() and dashboard is not None:
                    cv2.imwrite("dashboard_result.png", dashboard)
                print("[INFO] Da luu thanh cong cac anh ket qua: original_image.png, depth_map.png, segmentation_overlay.png, segmentation_color_mask.png, fusion_result.png, dashboard_result.png")
            print("="*70)
    except Exception as e:
        print(f"[WARNING] Khong the tu dong hoi luu file: {e}")
finally:
    # Dam bao luon dung cac luong khi chuong trinh ket thuc
    reader_thread.stop()
    inference_thread.stop()
    reader_thread.join(timeout=1.0)
    inference_thread.join(timeout=1.0)

cap.release()
cv2.destroyAllWindows()
print("[INFO] Chuong trinh ket thuc tot dep.")