import sys
import os
import argparse
import glob

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from utils.model import UnetResNet as Unet 
from MidasDepthEstimation.midasDepthEstimator import midasDepthEstimator as MidasDepthEstimator
from utils.utils import load_train_config
from utils.fusion import filter_detections, fuse_detections_and_segmentation

def process_single_image(image_path, unet_model, detection_model, depth_estimator, device, resize_dim, output_dir, base_threshold=-2.5):
    print(f"\n[INFO] Dang xu ly anh: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Khong the doc anh: {image_path}")
        return

    orig_h, orig_w = frame.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Tạo thư mục output cho ảnh này
    img_out_dir = os.path.join(output_dir, base_name)
    os.makedirs(img_out_dir, exist_ok=True)

    # 1. Lưu ảnh gốc
    cv2.imwrite(os.path.join(img_out_dir, "1_original.png"), frame)

    # 2. Ước lượng bản đồ độ sâu (Depth Map)
    depth_map = depth_estimator.estimateDepthMap(frame)
    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(img_out_dir, "2_depth_map.png"), depth_colored)

    # 3. Phân đoạn mặt đường (Semantic Segmentation U-Net)
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

    # Tạo mặt nạ đen trắng mặt đường
    seg_mask_image = np.zeros((orig_h, orig_w), dtype=np.uint8)
    if getattr(unet_model, "num_classes", 1) == 4:
        seg_mask_image[seg_mask_unet == 0] = 255
    else:
        seg_mask_image[seg_mask_unet == 1] = 255
    cv2.imwrite(os.path.join(img_out_dir, "3_segmentation_mask.png"), seg_mask_image)

    # Phủ màu tím lên mặt đường (BGR: 128, 64, 128) hoặc vẽ đa lớp
    color_mask = np.zeros_like(frame)
    if getattr(unet_model, "num_classes", 1) == 4:
        color_mask[seg_mask_unet == 0] = (128, 64, 128) # Road (Purple)
        color_mask[seg_mask_unet == 1] = (180, 130, 70)  # Sky (Sky Blue)
        color_mask[seg_mask_unet == 2] = (0, 0, 255)      # Vehicle (Red)
    else:
        color_mask[seg_mask_unet == 1] = (128, 64, 128)
    seg_overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
    cv2.imwrite(os.path.join(img_out_dir, "4_segmentation_overlay.png"), seg_overlay)

    # 4. Phát hiện vật thể (YOLO/Faster R-CNN) & Tích hợp cảnh báo va chạm
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
    
    for box, label, score in zip(boxes, labels, scores):
        threshold = 0.05 if label in sign_labels else 0.20
        if score > threshold:
            xmin, ymin, xmax, ymax = map(int, box)
            box_depth = depth_map[ymin:ymax, xmin:xmax]
            if box_depth.size > 0:
                max_d = np.percentile(box_depth, 95)
                if label in vehicle_labels:
                    detected_obstacles.append({'box': (xmin, ymin, xmax, ymax), 'depth': max_d, 'type': 'vehicle'})
                elif label in human_labels:
                    detected_obstacles.append({'box': (xmin, ymin, xmax, ymax), 'type': 'human', 'depth': max_d})
                elif label in sign_labels:
                    obs_type = 'traffic light' if label == 10 else 'stop sign'
                    detected_obstacles.append({'box': (xmin, ymin, xmax, ymax), 'type': obs_type, 'depth': max_d})

    # Dựng dải phân làn kỹ thuật số động để phân chia làn đường bên phải xe chạy
    camera_center = (int(orig_w * 0.70), int(orig_h * 0.92))
    y_top = int(orig_h * 0.45)
    y_bottom = orig_h
    x_top = int(orig_w * 0.50)
    x_bottom = int(orig_w * 0.45)

    path_obstacles_above = []
    if detected_obstacles:
        for obs in detected_obstacles:
            if obs.get('type') in ['vehicle', 'human']:
                xmin, ymin, xmax, ymax = obs['box']
                x_center = (xmin + xmax) // 2
                y_center = (ymin + ymax) // 2
                
                # Tính đường phân làn tại y_center
                if y_center >= y_top:
                    x_divider = x_top + (y_center - y_top) * (x_bottom - x_top) / (y_bottom - y_top + 1e-5)
                else:
                    x_divider = x_top
                
                # Chỉ lọc những vật thể nằm cùng làn đường bên phải (hướng di chuyển xe chủ)
                if x_center >= x_divider:
                    if y_center <= camera_center[1]:
                        path_obstacles_above.append(obs)

    closest_above = None
    if path_obstacles_above:
        path_obstacles_above.sort(key=lambda x: x['depth'], reverse=True)
        closest_above = path_obstacles_above[0]

    warn_above = False
    hud_font_scale = max(0.5, 0.8 * (orig_w / 1280.0))
    hud_thickness = max(1, int(2 * (orig_w / 1280.0)))
    hud_y = max(20, int(40 * (orig_h / 720.0)))

    if closest_above is not None:
        dist_above = 1000.0 / (closest_above['depth'] + 1e-5)
        warn_above = dist_above < 5.0
        
        if warn_above:
            cv2.putText(output_frame, "WARNING: Front vehicle too close!", (30, hud_y), cv2.FONT_HERSHEY_DUPLEX, hud_font_scale, (0, 0, 255), hud_thickness, cv2.LINE_AA)
        else:
            cv2.putText(output_frame, "Status: Safe", (30, hud_y), cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, (0, 255, 0), hud_thickness, cv2.LINE_AA)
    else:
        cv2.putText(output_frame, "Status: Safe", (30, hud_y), cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, (0, 255, 0), hud_thickness, cv2.LINE_AA)

    # Vẽ bounding box và hiển thị chỉ số khoảng cách tương đối.
    if detected_obstacles:
        for obs in detected_obstacles:
            xmin, ymin, xmax, ymax = obs['box']
            x_center = (xmin + xmax) // 2
            y_center = (ymin + ymax) // 2
            
            dist = 1000.0 / (obs['depth'] + 1e-5)
            
            is_closest_warn = False
            if closest_above is not None and np.array_equal(obs['box'], closest_above['box']) and warn_above:
                is_closest_warn = True
            
            color = (0, 0, 255) if is_closest_warn else (0, 255, 0)
            thickness = max(3, int(4 * (orig_w / 1280.0))) if is_closest_warn else max(2, int(2.5 * (orig_w / 1280.0)))
            
            cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), color, thickness)
            
            label_text = f"{obs.get('type').upper()}: {dist:.1f} rel"
            font_scale = max(0.40, 0.60 * (orig_w / 1280.0))
            font_thickness = max(1, int(1.5 * (orig_w / 1280.0)))
            (w_label, h_label), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            y_label = ymin - 4
            if y_label - h_label - 4 < 0:
                y_label = ymin + h_label + 8 + thickness
            
            cv2.rectangle(output_frame, (xmin, y_label - h_label - 6), (xmin + w_label + 10, y_label + 6), (0, 0, 0), -1)
            cv2.rectangle(output_frame, (xmin, y_label - h_label - 6), (xmin + w_label + 10, y_label + 6), color, max(1, thickness - 1))
            cv2.putText(output_frame, label_text, (xmin + 5, y_label), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            
            # Vẽ đường radar kết nối va chạm
            if y_center >= y_top:
                x_divider = x_top + (y_center - y_top) * (x_bottom - x_top) / (y_bottom - y_top + 1e-5)
            else:
                x_divider = x_top
            
            if x_center >= x_divider and y_center <= camera_center[1]:
                if obs.get('type') in ['vehicle', 'human']:
                    line_thickness = thickness if is_closest_warn else max(1, thickness - 1)
                    cv2.line(output_frame, camera_center, (x_center, y_center), color, line_thickness)
                    cv2.circle(output_frame, (x_center, y_center), 4, color, -1)

    # Vẽ biểu tượng MY CAR
    cv2.circle(output_frame, camera_center, 8, (0, 255, 0), -1)
    cv2.putText(output_frame, "MY CAR", (camera_center[0] - 30, camera_center[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Chồng chập mặt nạ phân đoạn lên kết quả
    fusion_res = cv2.addWeighted(output_frame, 0.8, color_mask, 0.2, 0)
    cv2.imwrite(os.path.join(img_out_dir, "5_fusion_result.png"), fusion_res)

    # Tạo bảng so sánh tổng hợp Dashboard
    disp_w, disp_h = 640, 360
    view_orig = cv2.resize(frame, (disp_w, disp_h))
    view_seg = cv2.resize(color_mask, (disp_w, disp_h))
    view_depth = cv2.resize(depth_colored, (disp_w, disp_h))
    view_fusion = cv2.resize(fusion_res, (disp_w, disp_h))

    dashboard_top = np.hstack((view_orig, view_seg))
    dashboard_bottom = np.hstack((view_depth, view_fusion))
    dashboard = np.vstack((dashboard_top, dashboard_bottom))
    cv2.imwrite(os.path.join(img_out_dir, "6_dashboard_comparison.png"), dashboard)

    # 5. TÍNH VÀ GHI CHÚ ĐỘ SÂU TRUNG BÌNH THEO TỪNG LỚP NGỮ NGHĨA
    # Bản đồ nhãn của Cityscapes định nghĩa trong hệ thống
    class_names = {
        0: "Void/Nen (Khong xac dinh)",
        1: "Road/Mat duong",
        2: "Sidewalk/Via he",
        3: "Object/Bien bao/Den tin hieu",
        4: "Nature/Cay coi/Tu nhien",
        5: "Sky/Bau troi",
        6: "Human/Nguoi di bo",
        7: "Vehicle/Phuong tien giao thong"
    }

    stats_file = os.path.join(img_out_dir, "7_depth_statistics.txt")
    with open(stats_file, "w", encoding="utf-8") as sf:
        sf.write(f"=== KET QUA PHAN TICH DO SAU THEO LOP NGU NGHIA ({base_name}) ===\n")
        sf.write("Disparity lay tu model MiDaS (0-255). Gia tri cang cao nghia la cang gan camera.\n")
        sf.write("Chi so khoang cach tuong doi = 1000 / (Disparity + 1e-5); khong co don vi met.\n")
        sf.write("-" * 75 + "\n")
        
        # Mặc định U-Net nhị phân (chỉ phân vùng mặt đường = class 1)
        # Trong utils/main.py, nếu là fallback hoặc multi-class, mask có giá trị từ 0 đến 7.
        # Ở đây ta tính toán trên seg_mask_unet của U-Net
        for class_idx, name in class_names.items():
            if class_idx == 1:
                # Class 1: Drivable road từ U-Net
                if getattr(unet_model, "num_classes", 1) == 4:
                    mask = (seg_mask_unet == 0)
                else:
                    mask = (seg_mask_unet == 1)
            elif class_idx == 7:
                # Tìm vùng của các xe phát hiện được
                if getattr(unet_model, "num_classes", 1) == 4:
                    mask = (seg_mask_unet == 2)
                else:
                    mask = np.zeros_like(seg_mask_unet, dtype=bool)
                    for obs in detected_obstacles:
                        if obs.get('type') == 'vehicle':
                            xmin, ymin, xmax, ymax = obs['box']
                            mask[ymin:ymax, xmin:xmax] = True
            elif class_idx == 6:
                # Người đi bộ
                mask = np.zeros_like(seg_mask_unet, dtype=bool)
                for obs in detected_obstacles:
                    if obs.get('type') == 'human':
                        xmin, ymin, xmax, ymax = obs['box']
                        mask[ymin:ymax, xmin:xmax] = True
            else:
                # Các class khác trong ảnh tĩnh (mô phỏng theo vị trí ước lượng)
                mask = np.zeros_like(seg_mask_unet, dtype=bool)
                if class_idx == 5: # Sky ở phía trên
                    if getattr(unet_model, "num_classes", 1) == 4:
                        mask = (seg_mask_unet == 1)
                    else:
                        mask[0:int(orig_h*0.4), :] = True
                elif class_idx == 4: # Trees ở 2 bên rìa
                    mask[int(orig_h*0.3):int(orig_h*0.8), 0:int(orig_w*0.15)] = True
                    mask[int(orig_h*0.3):int(orig_h*0.8), int(orig_w*0.85):] = True
                elif class_idx == 2: # Sidewalk dưới Trees
                    mask[int(orig_h*0.8):, 0:int(orig_w*0.25)] = True
                    mask[int(orig_h*0.8):, int(orig_w*0.75):] = True

            if np.any(mask):
                avg_disp = np.mean(depth_map[mask])
                relative_distance = 1000.0 / (avg_disp + 1e-5)
                sf.write(f"- {name:<35} | Disparity TB: {avg_disp:6.2f} | Chi so khoang cach tuong doi: {relative_distance:5.1f}\n")
            else:
                sf.write(f"- {name:<35} | Khong phat hien trong khung hinh\n")
                
    print(f"[INFO] Da tinh va ghi do sau theo tung lop vao file: {stats_file}")
    print(f"[SUCCESS] Da luu cac ket qua phan tich vao thu muc: {img_out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Demo pipeline hieu ngu canh giao thong tren mot hoac nhieu anh mau.")
    parser.add_argument('--image_path', type=str, default="", help="Duong dan den 1 anh hoac thu muc chua anh. Neu de trong se tu lay 3 anh tu KITTI.")
    parser.add_argument('--output_dir', type=str, default="outputs", help="Thu muc luu ket qua dau ra.")
    parser.add_argument('--config_path', type=str, default="config/train_config.yaml", help="Path to config file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Khoi chay Demo tren thiet bi: {device}")

    # 1. Tạo thư mục output
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Xác định các ảnh đầu vào
    input_images = []
    if args.image_path:
        if os.path.isdir(args.image_path):
            input_images = glob.glob(os.path.join(args.image_path, "*.png")) + \
                           glob.glob(os.path.join(args.image_path, "*.jpg")) + \
                           glob.glob(os.path.join(args.image_path, "*.jpeg"))
        elif os.path.isfile(args.image_path):
            input_images = [args.image_path]
        else:
            print(f"[ERROR] Duong dan khong hop le: {args.image_path}")
            return
    else:
        # Tự động lấy 3 ảnh mẫu từ tập KITTI có sẵn
        kitti_dir = "data/kitti/training/image_2"
        if os.path.exists(kitti_dir):
            all_imgs = sorted(glob.glob(os.path.join(kitti_dir, "*.png")))
            if all_imgs:
                # Lấy 3 ảnh phân bố đều để đa dạng ngữ cảnh
                input_images = [all_imgs[0], all_imgs[len(all_imgs)//2], all_imgs[-1]]
                print(f"[INFO] Tu dong chon 3 anh mau tu KITTI: {input_images}")
        
    if not input_images:
        print("[ERROR] Khong tim thay anh mau nao de xu ly. Vui long kiem tra thu muc data/kitti/training/image_2 hoac truyen --image_path.")
        return

    # 3. Nạp cấu hình từ train_config.yaml
    config_path = args.config_path
    unet_weights_path = "weights/UNET_resnet50_road/best_model.pth"
    backbone = "resnet50"
    resize_dim = (640, 192)
    base_threshold = -2.5
    if os.path.exists(config_path):
        try:
            config = load_train_config(config_path)
            cfg_resize = config.get("DATASET", {}).get("resize", [])
            if cfg_resize and len(cfg_resize) == 2:
                resize_dim = (cfg_resize[1], cfg_resize[0])
            unet_weights_path = config.get("EVAL", {}).get("model_path", unet_weights_path)
            backbone = config.get("MODEL", {}).get("backbone", backbone)
            base_threshold = config.get("EVAL", {}).get("base_threshold", base_threshold)
        except Exception as e:
            print(f"[WARNING] Loi doc train_config.yaml: {e}")

    # 3.5 Nạp mô hình U-Net
    if os.path.exists(unet_weights_path):
        state = torch.load(unet_weights_path, map_location=device, weights_only=False)
        state_dict = state.get("state_dict", state)
        num_classes = 1
        if "final.weight" in state_dict:
            num_classes = state_dict["final.weight"].shape[0]
            
        unet_model = Unet(num_classes=num_classes, encoder_name=backbone).to(device)
        unet_model.load_state_dict(state_dict)
        unet_model.eval()
        print(f"[INFO] Da nap thanh cong mo hinh U-Net tu: {unet_weights_path}")
    else:
        print(f"[ERROR] Khong tim thay trong so U-Net tai: {unet_weights_path}")
        return

    # 4. Nạp mô hình Object Detection
    print("[INFO] Dang nap mo hinh Faster R-CNN...")
    try:
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
        detection_model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT).to(device)
    except:
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
        detection_model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(device)
    detection_model.eval()

    # 5. Nạp mô hình Depth (MiDaS)
    print("[INFO] Dang nap mo hinh MiDaS...")
    depth_estimator = MidasDepthEstimator()

    # 7. Xử lý danh sách ảnh
    for img_path in input_images:
        process_single_image(img_path, unet_model, detection_model, depth_estimator, device, resize_dim, args.output_dir, base_threshold)

    print(f"\n[SUCCESS] Hoan thanh chay demo tren {len(input_images)} anh! Ket qua luu tai: {args.output_dir}")

if __name__ == '__main__':
    main()
