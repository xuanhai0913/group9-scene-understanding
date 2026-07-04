# README 6: Testing & Verification (Thử nghiệm và Đánh giá)

Tài liệu này hướng dẫn cách chạy thử nghiệm hệ thống và ghi nhận kết quả đánh giá mô hình.

---

## 1. Hướng dẫn chạy thử nghiệm thời gian thực (Real-time Pipeline)

Chạy chương trình điều phối chính để thực hiện nhận diện làn đường, ước lượng gần/xa tương đối và hiển thị cảnh báo minh họa trên video:
```powershell
python -m utils.main
```

### Hướng dẫn sử dụng:
*   Màn hình console sẽ xuất hiện danh sách lựa chọn video có sẵn trong thư mục `data/sample_videos`.
*   Bạn có thể nhập số tương ứng (ví dụ: `1`, `2`, `3`) hoặc trực tiếp kéo thả file video của bạn vào cửa sổ terminal rồi nhấn `Enter`.
*   Nhấn phím **`q`** tại màn hình video để thoát chương trình bất kỳ lúc nào.

---

## 2. Hướng dẫn đánh giá định lượng (Quantitative Evaluation)

Chạy script đánh giá mô hình U-Net trên 58 ảnh validation của tập dữ liệu KITTI:
```powershell
python eval.py --config_path config/train_config.yaml
```

### Kết quả đo đạc thực tế:
```
***** Prediction done in 332 sec.; IoU: 0.8335252373382963, Dice: 0.9064655509488333 ***** 

==================================================
***** CLASSIFICATION METRICS (Pixel-wise) *****
==================================================

Metrics:
  - Accuracy:  0.965134
  - Precision: 0.835061
  - Recall:    0.997551
  - F1-Score:  0.909102

Confusion Matrix:
                     Predicted Neg    Predicted Pos
Actual Neg (BG)           21361799           930279
Actual Pos (FG)              11564          4709858
--------------------------------------------------
```

*   **Chỉ số Accuracy:** **96.51%** (Tỷ lệ pixel đoán đúng nhãn trên ảnh).
*   **Chỉ số Recall:** **99.76%** (Độ phủ làn đường, chứng tỏ mô hình hạn chế tối đa việc bỏ sót vệt đường).
*   **Chỉ số IoU:** **83.35%** (Đạt tiêu chuẩn chất lượng cao đối với phân đoạn thời gian thực).

---

## 3. Tạo biểu đồ Heatmap Confusion Matrix cho báo cáo
Để tự sinh ra biểu đồ ma trận nhầm lẫn heatmap màu xanh phục vụ việc chèn vào slide báo cáo:
```powershell
python generate_heatmap.py
```
Hình ảnh kết quả sẽ được lưu trực tiếp tại file **`confusion_matrix_heatmap.png`** trong thư mục dự án.

---

## 4. Bảng kiểm thử theo từng Module (Testing Checklist)

Để đảm bảo tính đúng đắn khi vận hành hệ thống, dưới đây là danh sách kiểm thử (Checklist) cho từng Module cốt lõi:

| STT | Module | Kịch bản kiểm thử (Test Scenario) | Kết quả kỳ vọng (Expected Result) | Trạng thái |
|:---|:---|:---|:---|:---|
| 1 | **Data Loader & Preprocess** | Đọc dữ liệu ảnh/video đầu vào, thực hiện Resize về `(480, 270)` mỗi panel để tối ưu Full HD, chuẩn hóa và chuyển sang Tensor. | Ảnh được xử lý đúng kích thước, bộ nạp hoạt động đa luồng mượt mà, không gây nghẽn. | **Passed** |
| 2 | **Segmentation (U-Net)** | Đưa ảnh qua mô hình U-Net để dự đoán phân vùng mặt đường và phương tiện/con người. | Trả về mặt nạ phân đoạn 8 lớp. Đạt chỉ số IoU ~83.35% trên tập validation. | **Passed** |
| 3 | **Depth Estimation (MiDaS)** | Ước lượng độ sâu từ ảnh đơn sắc sử dụng MiDaS TFLite. | Trả về ma trận độ sâu disparity. Vùng ở gần xe có màu ấm, vùng ở xa có màu lạnh. | **Passed** |
| 4 | **Object Bounding Box (Contours)** | Trích xuất vị trí hộp bao (Bounding Box) trực tiếp từ mặt nạ phân đoạn màu đỏ (Vehicle) và màu hồng (Human) của U-Net bằng OpenCV. | Trả về danh sách tọa độ hộp bao của phương tiện/người đi đường chính xác mà không cần Faster R-CNN. | **Passed** |
| 5 | **Fusion & Relative Depth HUD** | Đồng bộ hóa hộp bao với giá trị depth tương ứng và tính chỉ số khoảng cách tương đối. | Gán ID và hiển thị khoảng cách động (ví dụ: `4.2 rel`) hợp lý cho từng đối tượng. | **Passed** |
| 6 | **Optical Flow Motion Check** | Đo độ trôi nền của hậu cảnh bằng Lucas-Kanade Optical Flow để phân loại Camera tĩnh/động. | Tự động nhận diện đúng loại Camera (CCTV cố định vs Dashcam di chuyển) và cấu hình chế độ Full Road/Split Road. | **Passed** |
| 7 | **Dynamic Lane Selector** | Kiểm tra xe gần nhất để tự động thay đổi hướng làn Ego lệch trái hoặc lệch phải. | Căn khớp làn chính xác theo hướng di chuyển thực tế (ví dụ xe máy đi bên trái dải phân cách ở video TP. HCM). | **Passed** |
| 8 | **HUD Collision Warning** | Kiểm tra chướng ngại vật cùng làn thỏa điều kiện cảnh báo heuristic (khoảng cách tương đối < 4.5). | Kích hoạt banner đỏ `COLLISION WARNING: Obstacle too close!`; không diễn giải thành mét. | **Passed** |
