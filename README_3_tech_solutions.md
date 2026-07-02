# README 3: Technical Solutions (Giải pháp công nghệ)

Tài liệu này giới thiệu kiến trúc các mô hình học máy và thư viện được sử dụng để xây dựng hệ thống.

---

## 1. Mô hình phân đoạn làn đường (Road Segmentation Model)
*   **Mạng nơ-ron:** Sử dụng kiến trúc **U-Net** cải tiến kết hợp với bộ mã hóa (Backbone) là mạng **ResNet-50** đã huấn luyện sẵn.
*   **Dataset:** Mô hình được tinh chỉnh và huấn luyện trên bộ dữ liệu **KITTI Road Dataset**.
*   **Tối ưu hóa:** Loại bỏ Test Time Augmentation (TTA) để tăng tốc độ suy luận gấp 1.5 lần và áp dụng kỹ thuật tinh chỉnh ngưỡng quyết định (Threshold Tuning) đạt hiệu năng tối ưu trên CPU.

## 2. Mô hình ước lượng độ sâu đơn sắc (Monocular Depth Model)
*   **Mạng nơ-ron:** Sử dụng mô hình **MiDaS v2.1 Small** (kiến trúc gọn nhẹ chuyên dụng cho thiết bị biên và CPU).
*   **Đặc điểm:** Cho phép tính toán chiều sâu từ 1 ảnh camera hành trình duy nhất mà không cần camera stereo hay cảm biến Lidar đắt tiền.

## 3. Mô hình phát hiện vật thể (Object Detection Model)
*   **Mạng nơ-ron:** Sử dụng bộ phát hiện vật thể **Faster R-CNN** (với backbone **MobileNetV3-Large** hoặc **ResNet50 FPN**) để khoanh vùng các hộp bao phương tiện (xe hơi, xe máy, xe bus, v.v.).
*   **Theo dõi:** Sử dụng bộ lọc Kalman Filter kết hợp IoU Tracker để theo dõi phương tiện ổn định qua các khung hình liên tục.

## 4. Công nghệ triển khai hệ thống
*   **Ngôn ngữ lập trình:** Python 3.8+
*   **Thư viện xử lý ảnh:** OpenCV (cv2) để đọc/ghi video, vẽ HUD và biến đổi hình ảnh.
*   **Deep Learning Framework:** PyTorch & Torchvision để chạy suy luận mô hình phân đoạn và mô hình độ sâu.
