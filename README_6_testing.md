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

Chạy script đánh giá mô hình U-Net trên 585 ảnh validation của tập dữ liệu Cityscapes:
```powershell
python eval.py --config_path config/train_config_cityscapes.yaml
```

### Kết quả đo đạc thực tế sau tối ưu hóa (Epoch 50):
*   **Chỉ số IoU tổng thể (Mean IoU):** **82.00%** (`0.819999`)
*   **Chỉ số Dice tổng thể (Mean Dice):** **89.78%** (`0.897798`)

---

### BẢNG CHỈ SỐ ĐÁNH GIÁ CHI TIẾT (TẬP TRUNG 3 LỚP CÔ YÊU CẦU)

| Lớp đối tượng | Accuracy | Precision | Recall | F1-Score (Dice) | Loại nhãn |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🛣️ **Đường (flat/road)** | 97.10% | 94.97% | 97.82% | **96.37%** | **Lớp chính (Yêu cầu)** |
| ☁️ **Bầu trời (sky)** | 99.49% | 92.54% | 95.89% | **94.19%** | **Lớp chính (Yêu cầu)** |
| 🚗 **Xe cộ (vehicle/car)** | 98.91% | 93.20% | 92.37% | **92.78%** | **Lớp chính (Yêu cầu)** |
| 🌳 Cây cối (nature) | 96.75% | 94.15% | 86.35% | 90.08% | Lớp phụ trợ |
| 🏢 Công trình (construction) | 94.59% | 91.48% | 82.18% | 86.58% | Lớp phụ trợ |
| 🚶 Con người (human) | 99.57% | 71.45% | 64.35% | 67.71% | Lớp phụ trợ |
| 🕳️ Nền void (void) | 95.68% | 98.76% | 47.38% | 64.03% | Lớp phụ trợ |
| 🚧 Cột/Biển báo (object) | 98.39% | 68.29% | 12.68% | 21.39% | Lớp phụ trợ |

> [!NOTE]
> Nhờ áp dụng **FocalDiceLoss** kết hợp **Hard Data Augmentations** và bộ **Trọng số Cực đoan (Class Weights: 5.0)**, cả 3 lớp chính cô yêu cầu đều đạt hiệu năng vượt trội **>92% F1-score**, khắc phục hoàn toàn tình trạng mất cân bằng mẫu và bỏ sót pixel (Recall tăng từ ~45% lên >92%).

---

### Chi tiết Confusion Matrix & Metrics của 3 Lớp Chính:

```
Class 1 (flat - Road):
  - Accuracy:  0.971015 | Precision: 0.949657 | Recall: 0.978174 | F1-Score: 0.963705
  Confusion Matrix:
                     Predicted Neg    Predicted Pos
  Actual Neg (BG)          101136432          3519268
  Actual Pos (FG)            1481289         66386531
--------------------------------------------------
Class 5 (sky - Bầu trời):
  - Accuracy:  0.994945 | Precision: 0.925393 | Recall: 0.958922 | F1-Score: 0.941859
  Confusion Matrix:
                     Predicted Neg    Predicted Pos
  Actual Neg (BG)          164588132           569459
  Actual Pos (FG)             302580          7063349
--------------------------------------------------
Class 7 (vehicle - Xe cộ):
  - Accuracy:  0.989142 | Precision: 0.932032 | Recall: 0.923654 | F1-Score: 0.927824
  Confusion Matrix:
                     Predicted Neg    Predicted Pos
  Actual Neg (BG)          158610307           878009
  Actual Pos (FG)             995182         12040022
```

---

## 3. Bảng kiểm thử theo từng Module (Testing Checklist)

Để đảm bảo tính đúng đắn khi vận hành hệ thống, dưới đây là danh sách kiểm thử (Checklist) cho từng Module cốt lõi:

| STT | Module | Kịch bản kiểm thử (Test Scenario) | Kết quả kỳ vọng (Expected Result) | Trạng thái |
|:---|:---|:---|:---|:---|
| 1 | **Data Loader & Preprocess** | Đọc dữ liệu ảnh/video đầu vào, thực hiện Resize về `(480, 270)` mỗi panel để tối ưu Full HD, chuẩn hóa và chuyển sang Tensor. | Ảnh được xử lý đúng kích thước, bộ nạp hoạt động đa luồng mượt mà, không gây nghẽn. | **Passed** |
| 2 | **Segmentation (U-Net)** | Đưa ảnh qua mô hình U-Net để dự đoán phân vùng mặt đường và phương tiện/con người. | Trả về mặt nạ phân đoạn 8 lớp. Đạt chỉ số IoU ~82.00% trên tập validation Cityscapes. | **Passed** |
| 3 | **Depth Estimation (MiDaS)** | Ước lượng độ sâu từ ảnh đơn sắc sử dụng MiDaS TFLite. | Trả về ma trận độ sâu disparity. Vùng ở gần xe có màu ấm, vùng ở xa có màu lạnh. | **Passed** |
| 4 | **Object Bounding Box (Contours)** | Trích xuất vị trí hộp bao (Bounding Box) trực tiếp từ mặt nạ phân đoạn màu đỏ (Vehicle) và màu hồng (Human) của U-Net bằng OpenCV. | Trả về danh sách tọa độ hộp bao của phương tiện/người đi đường chính xác mà không cần Faster R-CNN. | **Passed** |
| 5 | **Fusion & Relative Depth HUD** | Đồng bộ hóa hộp bao với giá trị depth tương ứng và tính chỉ số khoảng cách tương đối. | Gán ID và hiển thị khoảng cách động (ví dụ: `4.2 rel`) hợp lý cho từng đối tượng. | **Passed** |
| 6 | **Optical Flow Motion Check** | Đo độ trôi nền của hậu cảnh bằng Lucas-Kanade Optical Flow để phân loại Camera tĩnh/động. | Tự động nhận diện đúng loại Camera (CCTV cố định vs Dashcam di chuyển) và cấu hình chế độ Full Road/Split Road. | **Passed** |
| 7 | **Dynamic Lane Selector** | Kiểm tra xe gần nhất để tự động thay đổi hướng làn Ego lệch trái hoặc lệch phải. | Căn khớp làn chính xác theo hướng di chuyển thực tế (ví dụ xe máy đi bên trái dải phân cách ở video TP. HCM). | **Passed** |
| 8 | **HUD Collision Warning** | Kiểm tra chướng ngại vật cùng làn thỏa điều kiện cảnh báo heuristic (khoảng cách tương đối < 4.5). | Kích hoạt banner đỏ `COLLISION WARNING: Obstacle too close!`; không diễn giải thành mét. | **Passed** |
