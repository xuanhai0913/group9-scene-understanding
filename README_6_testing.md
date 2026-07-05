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
*   **Chỉ số IoU tổng thể (Mean IoU):** **47.94%** (`0.479395`)
*   **Chỉ số Dice tổng thể (Mean Dice):** **64.36%** (`0.643593`)

---

### BẢNG CHỈ SỐ ĐÁNH GIÁ CHI TIẾT (TẬP TRUNG 3 LỚP CHÍNH)

| Lớp đối tượng | Accuracy | Precision | Recall | F1-Score (Dice) | Loại nhãn |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🛣️ **Đường (flat/road)** | 80.50% | 79.79% | 67.54% | **73.15%** | **Lớp chính (Yêu cầu)** |
| ☁️ **Bầu trời (sky)** | 98.13% | 85.44% | 67.86% | **75.64%** | **Lớp chính (Yêu cầu)** |
| 🚗 **Xe cộ (vehicle/car)** | 95.37% | 67.04% | 76.12% | **71.29%** | **Lớp chính (Yêu cầu)** |
| 🌳 Cây cối (nature) | 95.17% | 92.06% | 78.46% | 84.72% | Lớp phụ trợ |
| 🏢 Công trình (construction) | 76.30% | 46.82% | 86.12% | 60.66% | Lớp phụ trợ |
| 🚶 Con người (human) | 99.33% | 52.86% | 38.93% | 44.84% | Lớp phụ trợ |
| 🕳️ Nền void (void) | 91.67% | 47.94% | 30.30% | 37.13% | Lớp phụ trợ |
| 🚧 Cột/Biển báo (object) | 98.32% | 75.53% | 3.73% | 7.10% | Lớp phụ trợ |

> [!NOTE]
> Nhờ áp dụng kỹ thuật **Hậu xử lý hình thái học phân tách lớp (Morphology Split)** và **Tối ưu hóa ngưỡng quyết định (Threshold Tuning)** động, cả 3 lớp chính đều đạt hiệu năng vượt trội. Lớp Bầu trời đạt F1-Score **75.64%** (tăng +15.31%) và Đường đi đạt **73.15%** (tăng +8.99%), trong khi Xe cộ vẫn bảo toàn ở mức cao nhất **71.29%**.

---

### Chi tiết Confusion Matrix & Metrics của 3 Lớp Chính:

```
Class 1 (flat - Road):
  - Accuracy:  0.804991 | Precision: 0.797855 | Recall: 0.675395 | F1-Score: 0.731536
  Confusion Matrix:
                     Predicted Neg    Predicted Pos
  Actual Neg (BG)           93042284         11613416
  Actual Pos (FG)           22030206         45837614
--------------------------------------------------
Class 5 (sky - Bầu trời):
  - Accuracy:  0.981340 | Precision: 0.854384 | Recall: 0.678610 | F1-Score: 0.756420
  Confusion Matrix:
                     Predicted Neg    Predicted Pos
  Actual Neg (BG)          164305660           851931
  Actual Pos (FG)            2367335          4998594
--------------------------------------------------
Class 7 (vehicle - Xe cộ):
  - Accuracy:  0.953678 | Precision: 0.670381 | Recall: 0.761175 | F1-Score: 0.712899
  Confusion Matrix:
                     Predicted Neg    Predicted Pos
  Actual Neg (BG)          154609743          4878573
  Actual Pos (FG)            3113136          9922068
```

---

## 3. Bảng kiểm thử theo từng Module (Testing Checklist)

Để đảm bảo tính đúng đắn khi vận hành hệ thống, dưới đây là danh sách kiểm thử (Checklist) cho từng Module cốt lõi:

| STT | Module | Kịch bản kiểm thử (Test Scenario) | Kết quả kỳ vọng (Expected Result) | Trạng thái |
|:---|:---|:---|:---|:---|
| 1 | **Data Loader & Preprocess** | Đọc dữ liệu ảnh/video đầu vào, thực hiện Resize về `(480, 270)` mỗi panel để tối ưu Full HD, chuẩn hóa và chuyển sang Tensor. | Ảnh được xử lý đúng kích thước, bộ nạp hoạt động đa luồng mượt mà, không gây nghẽn. | **Passed** |
| 2 | **Segmentation (U-Net)** | Đưa ảnh qua mô hình U-Net để dự đoán phân vùng mặt đường và phương tiện/con người. | Trả về mặt nạ phân đoạn 8 lớp. Đạt chỉ số IoU ~47.94% trên tập validation Cityscapes. | **Passed** |
| 3 | **Depth Estimation (MiDaS)** | Ước lượng độ sâu từ ảnh đơn sắc sử dụng MiDaS TFLite. | Trả về ma trận độ sâu disparity. Vùng ở gần xe có màu ấm, vùng ở xa có màu lạnh. | **Passed** |
| 4 | **Object Bounding Box (Contours)** | Trích xuất vị trí hộp bao (Bounding Box) trực tiếp từ mặt nạ phân đoạn màu đỏ (Vehicle) và màu hồng (Human) của U-Net bằng OpenCV. | Trả về danh sách tọa độ hộp bao của phương tiện/người đi đường chính xác mà không cần Faster R-CNN. | **Passed** |
| 5 | **Fusion & Relative Depth HUD** | Đồng bộ hóa hộp bao với giá trị depth tương ứng và tính chỉ số khoảng cách tương đối. | Gán ID và hiển thị khoảng cách động (ví dụ: `4.2 rel`) hợp lý cho từng đối tượng. | **Passed** |
| 6 | **Optical Flow Motion Check** | Đo độ trôi nền của hậu cảnh bằng Lucas-Kanade Optical Flow để phân loại Camera tĩnh/động. | Tự động nhận diện đúng loại Camera (CCTV cố định vs Dashcam di chuyển) và cấu hình chế độ Full Road/Split Road. | **Passed** |
| 7 | **Dynamic Lane Selector** | Kiểm tra xe gần nhất để tự động thay đổi hướng làn Ego lệch trái hoặc lệch phải. | Căn khớp làn chính xác theo hướng di chuyển thực tế (ví dụ xe máy đi bên trái dải phân cách ở video TP. HCM). | **Passed** |
| 8 | **HUD Collision Warning** | Kiểm tra chướng ngại vật cùng làn thỏa điều kiện cảnh báo heuristic (khoảng cách tương đối < 4.5). | Kích hoạt banner đỏ `COLLISION WARNING: Obstacle too close!`; không diễn giải thành mét. | **Passed** |
