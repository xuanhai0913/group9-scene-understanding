import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import os

from utils.model import UnetResNet as Unet 
from MidasDepthEstimation.midasDepthEstimator import midasDepthEstimator as MidasDepthEstimator
from utils.utils import load_train_config

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Demo single frame")
    parser.add_argument('--config_path', type=str, default="config/train_config.yaml", help="Path to config file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Khoi chay demo single frame tren: {device}")

    # 1. KHOI TAO CAC DUONG DAN MO HINH
    unet_weights_path = "weights/UNET_resnet50_road/best_model.pth"
    config_path = args.config_path
    
    backbone = "resnet50"
    resize_dim = (640, 192) # width, height (from [192, 640])
    base_threshold = -2.5
    
    if os.path.exists(config_path):
        try:
            config = load_train_config(config_path)
            backbone = config.get("MODEL", {}).get("backbone", "resnet50")
            cfg_resize = config.get("DATASET", {}).get("resize", [])
            if cfg_resize and len(cfg_resize) == 2:
                resize_dim = (cfg_resize[1], cfg_resize[0])
            unet_weights_path = config.get("EVAL", {}).get("model_path", unet_weights_path)
            base_threshold = config.get("EVAL", {}).get("base_threshold", base_threshold)
        except Exception as e:
            print(f"[WARNING] Loi doc file train_config: {e}")

    # 2. NAP MO HINH U-NET
    if os.path.exists(unet_weights_path):
        state = torch.load(unet_weights_path, map_location=device, weights_only=False)
        state_dict = state.get("state_dict", state)
        num_classes = 1
        if "final.weight" in state_dict:
            num_classes = state_dict["final.weight"].shape[0]
            
        unet_model = Unet(num_classes=num_classes, encoder_name=backbone).to(device)
        unet_model.load_state_dict(state_dict)
        unet_model.eval()
        print(f"[INFO] Da nap file trong so U-Net {unet_weights_path} thanh cong.")
    else:
        print(f"[ERROR] Khong tim thay checkpoint U-Net tai: {unet_weights_path}")
        return

    # 3. NAP MO HINH OBJECT DETECTION
    print("[INFO] Dang nap mo hinh Faster R-CNN...")
    try:
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
        detection_model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT).to(device)
    except:
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
        detection_model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(device)
    detection_model.eval()

    # 4. NAP MO HINH DEPTH (MIDAS)
    print("[INFO] Dang nap mo hinh MiDaS TFLite...")
    depth_estimator = MidasDepthEstimator()

    # 5. MO VIDEO KIEU MAU DE LAY 1 FRAME
    video_path = "data/sample_videos/vecteezy_motorbikes-and-cars-traffic-on-hanoi-highway-vietnam_30520411.mov"
    if not os.path.exists(video_path):
        # Tim bat ky video nao neu khong co video tren
        import glob
        video_files = glob.glob("data/sample_videos/*.mp4") + glob.glob("data/sample_videos/*.mov")
        if video_files:
            video_path = video_files[0]
        else:
            print("[ERROR] Khong tim thay video nao trong data/sample_videos/")
            return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Khong the mo video: {video_path}")
        return

    # Lay frame thu 30 de co goc dep va vach ke/xe ro rang
    frame = None
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            break
    cap.release()

    if frame is None:
        print("[ERROR] Khong doc duoc frame tu video")
        return

    orig_h, orig_w = frame.shape[:2]

    # --- SAVE 1. ORIGINAL IMAGE ---
    cv2.imwrite("original_image.png", frame)
    print("[INFO] Da luu original_image.png")

    # 6. DU DOAN DEPTH MAP
    depth_map = depth_estimator.estimateDepthMap(frame)
    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    
    # --- SAVE 4. DEPTH MAP ---
    cv2.imwrite("depth_map.png", depth_colored)
    print("[INFO] Da luu depth_map.png")

    # 7. DU DOAN SEMANTIC SEGMENTATION (U-NET)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seg_resized = cv2.resize(frame_rgb, resize_dim) 
    unet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    seg_tensor = unet_transform(seg_resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        seg_output = unet_model(seg_tensor)
        if seg_output.shape[1] == 1:
            seg_mask = (seg_output > base_threshold).long().squeeze(0).squeeze(0).cpu().numpy()
        else:
            seg_mask = torch.argmax(seg_output, dim=1).squeeze(0).cpu().numpy()

    seg_mask_unet = cv2.resize(seg_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Tao mat na den trang cho segmentation mask
    # Mat duong se co mau trang (255), nen mau den (0)
    seg_mask_image = np.zeros((orig_h, orig_w), dtype=np.uint8)
    seg_mask_image[seg_mask_unet == 1] = 255
    
    # --- SAVE 2. SEGMENTATION MASK ---
    cv2.imwrite("segmentation_mask.png", seg_mask_image)
    print("[INFO] Da luu segmentation_mask.png")

    # Phủ màu tím lên mặt đường (BGR: 128, 64, 128)
    color_mask = np.zeros_like(frame)
    color_mask[seg_mask_unet == 1] = (128, 64, 128)
    seg_overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
    
    # --- SAVE 3. SEGMENTATION OVERLAY ---
    cv2.imwrite("segmentation_overlay.png", seg_overlay)
    print("[INFO] Da luu segmentation_overlay.png")

    # 8. PHAT HIEN VAT THE & FUSION LOGIC (CANH BAO VA CHAM)
    det_tensor = transforms.ToTensor()(frame_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = detection_model(det_tensor)[0]
    
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    vehicle_labels = {2, 3, 4, 6, 8} # car, motorcycle, bus, truck
    human_labels = {1}
    sign_labels = {10, 13}
    detected_obstacles = []
    
    output_frame = frame.copy()
    
    # Loc va luu cac vat the co nhan dien duoc
    for box, label, score in zip(boxes, labels, scores):
        # Ha nguong tin cay cho den tin hieu va bien bao vi chung rat nho va xa
        threshold = 0.05 if label in sign_labels else 0.20
        if score > threshold:
            xmin, ymin, xmax, ymax = map(int, box)
            box_depth = depth_map[ymin:ymax, xmin:xmax]
            if box_depth.size > 0:
                max_d = np.percentile(box_depth, 95)
                if label in vehicle_labels:
                    detected_obstacles.append({'box': (xmin, ymin, xmax, ymax), 'depth': max_d, 'type': 'vehicle'})
                elif label in human_labels:
                    detected_obstacles.append({'box': (xmin, ymin, xmax, ymax), 'depth': max_d, 'type': 'human'})
                elif label in sign_labels:
                    obs_type = 'traffic light' if label == 10 else 'stop sign'
                    detected_obstacles.append({'box': (xmin, ymin, xmax, ymax), 'depth': max_d, 'type': obs_type})

    camera_center = (int(orig_w * 0.70), int(orig_h * 0.92))
    y_top = int(orig_h * 0.45)
    y_bottom = orig_h
    x_top = int(orig_w * 0.50)
    x_bottom = int(orig_w * 0.45)

    path_obstacles_above = []
    if detected_obstacles:
        for obs in detected_obstacles:
            # Loai tru bien bao va den giao thong khoi canh bao va cham vi chung khong can tro duong xe chay
            if obs.get('type') in ['vehicle', 'human']:
                xmin, ymin, xmax, ymax = obs['box']
                x_center = (xmin + xmax) // 2
                y_center = (ymin + ymax) // 2
                
                if y_center >= y_top:
                    x_divider = x_top + (y_center - y_top) * (x_bottom - x_top) / (y_bottom - y_top + 1e-5)
                else:
                    x_divider = x_top
                
                if x_center >= x_divider:
                    if y_center <= camera_center[1]:
                        path_obstacles_above.append(obs)

    closest_above = None
    if path_obstacles_above:
        path_obstacles_above.sort(key=lambda x: x['depth'], reverse=True)
        closest_above = path_obstacles_above[0]

    # Hien thi thong tin va canh bao va cham
    warn_above = False
    hud_font_scale = max(0.6, 0.9 * (orig_w / 1280.0))
    hud_thickness = max(1, int(3 * (orig_w / 1280.0)))
    hud_y = max(20, int(50 * (orig_h / 720.0)))

    if closest_above is not None:
        dist_above = 1000.0 / (closest_above['depth'] + 1e-5)
        warn_above = dist_above < 5.0
        
        if warn_above:
            cv2.putText(output_frame, "WARNING: Front vehicle too close!", (30, hud_y), cv2.FONT_HERSHEY_DUPLEX, hud_font_scale, (0, 0, 255), hud_thickness, cv2.LINE_AA)
        else:
            cv2.putText(output_frame, "Status: Safe", (30, hud_y), cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, (0, 255, 0), hud_thickness, cv2.LINE_AA)
    else:
        cv2.putText(output_frame, "Status: Safe", (30, hud_y), cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, (0, 255, 0), hud_thickness, cv2.LINE_AA)

    # Ve tat ca chuong ngai vat va khoang cach
    if detected_obstacles:
        for obs in detected_obstacles:
            xmin, ymin, xmax, ymax = obs['box']
            x_center = (xmin + xmax) // 2
            y_center = (ymin + ymax) // 2
            
            # Chỉ số khoảng cách tương đối, không phải khoảng cách theo mét.
            dist = 1000.0 / (obs['depth'] + 1e-5)
            
            # Kiem tra xem co phai la vat the dang canh bao va cham khong
            is_closest_warn = False
            if closest_above is not None and np.array_equal(obs['box'], closest_above['box']) and warn_above:
                is_closest_warn = True
            
            # Mau sac cho vat the: Do neu la vat de doa va cham gan nhat, Nguoc lai la Xanh la
            color = (0, 0, 255) if is_closest_warn else (0, 255, 0)
            thickness = max(3, int(4 * (orig_w / 1280.0))) if is_closest_warn else max(2, int(2.5 * (orig_w / 1280.0)))
            
            # Ve hop bao quanh xe/nguoi
            cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), color, thickness)
            
            # Nhan loai vat the + chi so khoang cach tuong doi
            label_text = f"{obs.get('type', 'vehicle').upper()}: {dist:.1f} rel"
            font_scale = max(0.45, 0.65 * (orig_w / 1280.0))
            font_thickness = max(1, int(2 * (orig_w / 1280.0)))
            (w_label, h_label), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # Vi tri dat nhan (tinh toan de khong bi chong hoac ra ngoai anh)
            y_label = ymin - 4
            if y_label - h_label - 4 < 0:
                y_label = ymin + h_label + 8 + thickness
            
            # Ve hop nen chu mau den tuyet doi voi vien cung mau canh bao de de doc
            cv2.rectangle(output_frame, (xmin, y_label - h_label - 6), (xmin + w_label + 10, y_label + 6), (0, 0, 0), -1)
            cv2.rectangle(output_frame, (xmin, y_label - h_label - 6), (xmin + w_label + 10, y_label + 6), color, max(1, thickness - 1))
            
            # Ve chu mau trang sieu sac net su dung cv2.LINE_AA
            cv2.putText(output_frame, label_text, (xmin + 5, y_label), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            
            # Tinh toan duong phan lan tai y_center de quyet dinh ve duong radar hay khong
            if y_center >= y_top:
                x_divider = x_top + (y_center - y_top) * (x_bottom - x_top) / (y_bottom - y_top + 1e-5)
            else:
                x_divider = x_top
            
            # Neu nam trong lan duong cua minh (ben phai x_divider va phia tren camera_center)
            if x_center >= x_divider and y_center <= camera_center[1]:
                # Ve duong noi radar den xe
                line_thickness = thickness if is_closest_warn else max(1, thickness - 1)
                cv2.line(output_frame, camera_center, (x_center, y_center), color, line_thickness)
                cv2.circle(output_frame, (x_center, y_center), 4, color, -1)

    # Ve mui xe "MY CAR"
    cv2.circle(output_frame, camera_center, 8, (0, 255, 0), -1)
    cv2.putText(output_frame, "MY CAR", (camera_center[0] - 30, camera_center[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Chồng chập mặt nạ phân đoạn U-Net vào output_frame để tạo Fusion kết quả hoàn chỉnh
    fusion_res = cv2.addWeighted(output_frame, 0.8, color_mask, 0.2, 0)
    
    # --- SAVE 5. FUSION RESULT ---
    cv2.imwrite("fusion_result.png", fusion_res)
    print("[INFO] Da luu fusion_result.png")
    
    print("\n[SUCCESS] Successfully generated all 5 demo images in the project root directory!")

if __name__ == '__main__':
    main()
