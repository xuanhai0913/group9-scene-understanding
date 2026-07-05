import cv2
import numpy as np

def compute_overlap(box1, box2):
    """Compute Intersection over Union (IoU) and Intersection over Minimum Area (IoMin)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0, 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    iou = intersection / union if union > 0 else 0.0
    min_area = min(area1, area2)
    iomin = intersection / min_area if min_area > 0 else 0.0

    return iou, iomin

def filter_detections(boxes, labels, scores, orig_w, orig_h, scale_x_det, scale_y_det, depth_map):
    """Filters object detections using same-class NMS and cross-class Rider Merge suppression."""
    vehicle_labels = {2, 3, 4, 6, 8}
    human_labels = {1}
    sign_labels = {10, 13}
    
    detected_obstacles = []
    
    # Pass 1: Vehicles and Signs (NMS)
    for box, label, score in zip(boxes, labels, scores):
        threshold = 0.30 if label in sign_labels else 0.20
        if score > threshold:
            xmin = int(box[0] * scale_x_det)
            ymin = int(box[1] * scale_y_det)
            xmax = int(box[2] * scale_x_det)
            ymax = int(box[3] * scale_y_det)
            if ymin > int(orig_h * 0.90) and label in vehicle_labels:
                continue
                
            if label in vehicle_labels:
                obs_type = 'vehicle'
            elif label in sign_labels:
                obs_type = 'traffic light' if label == 10 else 'stop sign'
            else:
                continue
                
            overlap = False
            for existing in detected_obstacles:
                if existing['type'] == obs_type:
                    iou, iomin = compute_overlap((xmin, ymin, xmax, ymax), existing['box'])
                    if obs_type in ['traffic light', 'stop sign']:
                        if iou > 0.10 or iomin > 0.35:
                            overlap = True
                            break
                    else:
                        if iou > 0.20 or iomin > 0.45:
                            overlap = True
                            break
            if overlap:
                continue
                
            box_depth = depth_map[ymin:ymax, xmin:xmax]
            if box_depth.size > 0:
                max_d = np.percentile(box_depth, 95)
                detected_obstacles.append({
                    'box': (xmin, ymin, xmax, ymax),
                    'depth': max_d,
                    'type': obs_type,
                    'label': label
                })
                
    # Pass 2: Humans (with Rider Merge suppression against existing motorcycles/bicycles)
    for box, label, score in zip(boxes, labels, scores):
        if label in human_labels and score > 0.20:
            xmin = int(box[0] * scale_x_det)
            ymin = int(box[1] * scale_y_det)
            xmax = int(box[2] * scale_x_det)
            ymax = int(box[3] * scale_y_det)
            
            overlap = False
            for existing in detected_obstacles:
                if existing['type'] == 'human':
                    iou, iomin = compute_overlap((xmin, ymin, xmax, ymax), existing['box'])
                    if iou > 0.20 or iomin > 0.45:
                        overlap = True
                        break
                # Rider merge: suppress human if they overlap significantly with a motorcycle (4) or bicycle (2)
                elif existing['type'] == 'vehicle' and existing.get('label') in {2, 4}:
                    iou, iomin = compute_overlap((xmin, ymin, xmax, ymax), existing['box'])
                    if iomin > 0.55 or iou > 0.25:
                        overlap = True
                        break
            if overlap:
                continue
                
            box_depth = depth_map[ymin:ymax, xmin:xmax]
            if box_depth.size > 0:
                max_d = np.percentile(box_depth, 95)
                detected_obstacles.append({
                    'box': (xmin, ymin, xmax, ymax),
                    'depth': max_d,
                    'type': 'human',
                    'label': label
                })
                
    return detected_obstacles

def fuse_detections_and_segmentation(detections, seg_mask_full, depth_map, use_fallback_detection, num_classes=8):
    """
    Kết hợp mô hình nhận diện vật thể (Object Detection) và phân đoạn mặt nạ (Semantic Segmentation):
    - Nếu là mô hình phân đoạn nhị phân (chỉ có đường, num_classes == 1 hoặc fallback):
      Không lọc các phát hiện vật thể (tránh làm mất xe/người), chỉ lọc nếu vật thể nằm hoàn toàn trên bầu trời.
    - Nếu là mô hình đa lớp (num_classes > 1):
      1. Lọc nhiễu (False Positive Filtering) dựa trên sự trùng khớp nhãn phân đoạn.
      2. Phục hồi vật thể bị bỏ sót (Backup Recovery) từ mặt nạ phân đoạn U-Net.
    """
    fused_detections = []
    is_multiclass = (not use_fallback_detection) and (num_classes > 1)
    
    # Xác định chỉ số nhãn dựa trên số lượng lớp của U-Net
    road_class = 0 if num_classes == 4 else 1
    sky_class = 1 if num_classes == 4 else 5
    vehicle_class = 2 if num_classes == 4 else 7
    human_class = 6  # Không có trong U-Net 4 lớp
    sign_class = 3
    
    # 1. Lọc nhiễu
    for det in detections:
        box = det['box']
        xmin, ymin, xmax, ymax = box
        
        box_seg = seg_mask_full[ymin:ymax, xmin:xmax]
        if box_seg.size == 0:
            continue
            
        box_area = box_seg.size
        obs_type = det['type']
        
        if not is_multiclass:
            # Với mô hình nhị phân hoặc giả lập, chỉ lọc nếu hộp bao nằm phần lớn ở bầu trời (Class 5)
            sky_pixels = np.sum(box_seg == sky_class)
            if sky_pixels / box_area > 0.85:
                continue
            fused_detections.append(det)
            continue
            
        # Tính toán tỷ lệ cho mô hình đa lớp thực tế
        if obs_type == 'vehicle':
            matching_pixels = np.sum((box_seg == vehicle_class) | (box_seg == road_class))
            ratio = matching_pixels / box_area
            if ratio < 0.12:
                continue
        elif obs_type == 'human':
            if num_classes == 4:
                # Trong U-Net 4 lớp, human là background (class 3) nên chỉ cần check nằm trên đường (class 0)
                matching_pixels = np.sum(box_seg == road_class)
            else:
                matching_pixels = np.sum((box_seg == human_class) | (box_seg == road_class))
            ratio = matching_pixels / box_area
            if ratio < 0.08:
                continue
        elif obs_type in ['traffic light', 'stop sign']:
            if num_classes == 4:
                # Trong U-Net 4 lớp, biển báo thuộc background (class 3) nên không cần lọc theo mask
                ratio = 1.0
            else:
                matching_pixels = np.sum(box_seg == sign_class)
                ratio = matching_pixels / box_area
            if ratio < 0.03:
                continue
                
        fused_detections.append(det)
        
    # 2. Khôi phục vật thể bị bỏ sót (chỉ áp dụng cho đa lớp thực tế)
    if is_multiclass:
        # Khôi phục Xe (Vehicle)
        vehicle_mask = (seg_mask_full == vehicle_class).astype(np.uint8)
        # Sử dụng phép toán hình thái học (morphological operations) để nối các vùng phân mảnh và lọc nhiễu
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        vehicle_mask_cleaned = cv2.morphologyEx(vehicle_mask, cv2.MORPH_CLOSE, kernel)
        vehicle_mask_cleaned = cv2.morphologyEx(vehicle_mask_cleaned, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(vehicle_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sky_cutoff = int(depth_map.shape[0] * 0.48)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 350: # Đặt ở mức 350 để lọc nhiễu tốt hơn trong khi vẫn bắt được ô tô/xe máy ở xa
                x, y, w, h = cv2.boundingRect(contour)
                # Bỏ qua các khung bao nằm trên đường chân trời (trong tán cây, bầu trời)
                if y + h < sky_cutoff:
                    continue
                overlap = False
                for det in fused_detections:
                    if det['type'] == 'vehicle':
                        iou, _ = compute_overlap((x, y, x + w, y + h), det['box'])
                        if iou > 0.25:
                            overlap = True
                            break
                if not overlap:
                    box_depth = depth_map[y:y+h, x:x+w]
                    if box_depth.size > 0:
                        max_d = np.percentile(box_depth, 95)
                        fused_detections.append({
                            'box': (x, y, x + w, y + h),
                            'depth': max_d,
                            'type': 'vehicle',
                            'fused_backup': True
                        })
                        
        # Khôi phục Người (Human) - Chỉ khi mô hình U-Net hỗ trợ (num_classes != 4)
        if num_classes != 4:
            human_mask = (seg_mask_full == human_class).astype(np.uint8)
            # Áp dụng morphological operations cho mặt nạ người đi bộ
            human_mask_cleaned = cv2.morphologyEx(human_mask, cv2.MORPH_CLOSE, kernel)
            human_mask_cleaned = cv2.morphologyEx(human_mask_cleaned, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(human_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 300: # Đặt ở mức 300 để lọc bỏ các mảng nhiễu vỉa hè bị nhận nhầm thành người
                    x, y, w, h = cv2.boundingRect(contour)
                    # Bỏ qua người đi bộ phát hiện nhầm trên tán cây/bầu trời
                    if y + h < sky_cutoff:
                        continue
                    overlap = False
                    for det in fused_detections:
                        if det['type'] == 'human':
                            iou, _ = compute_overlap((x, y, x + w, y + h), det['box'])
                            if iou > 0.25:
                                overlap = True
                                break
                    if not overlap:
                        box_depth = depth_map[y:y+h, x:x+w]
                        if box_depth.size > 0:
                            max_d = np.percentile(box_depth, 95)
                            fused_detections.append({
                                
                                'box': (x, y, x + w, y + h),
                                'depth': max_d,
                                'type': 'human',
                                'fused_backup': True
                            })
                            
    return fused_detections
