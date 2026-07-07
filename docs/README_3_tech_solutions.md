# README 3: Technical Solutions (Giải pháp công nghệ)

Tài liệu này giới thiệu kiến trúc các mô hình học máy và thư viện được sử dụng để xây dựng hệ thống.

---

## 1. Mô hình phân đoạn làn đường (Road Segmentation Model)
*   **Mạng nơ-ron:** Sử dụng kiến trúc **U-Net** cải tiến kết hợp với bộ mã hóa (Backbone) là mạng **ResNet-50** đã huấn luyện sẵn.
*   **Dataset:** Mô hình được tinh chỉnh và huấn luyện trên bộ dữ liệu **Cityscapes Dataset** (8 lớp thực tế giao thông đô thị).
*   **Tối ưu hóa:** Loại bỏ Test Time Augmentation (TTA) để tăng tốc độ suy luận gấp 1.5 lần và áp dụng kỹ thuật tinh chỉnh ngưỡng quyết định (Threshold Tuning) đạt hiệu năng tối ưu trên CPU.

## 2. Mô hình ước lượng độ sâu đơn sắc (Monocular Depth Model)
*   **Mạng nơ-ron:** Sử dụng mô hình **MiDaS v2.1 Small** (kiến trúc gọn nhẹ chuyên dụng cho thiết bị biên và CPU).
*   **Đặc điểm:** Cho phép tính toán chiều sâu từ 1 ảnh camera hành trình duy nhất mà không cần camera stereo hay cảm biến Lidar đắt tiền.

## 3. Giải pháp trích xuất hộp bao lai (Hybrid Object Bounding Box Extraction)
*   **Thuật toán đường bao (OpenCV Contours):** Sử dụng thuật toán tìm đường biên (`cv2.findContours`) trực tiếp trên kết quả phân đoạn lớp Phương tiện (Vehicle) và Con người (Human) của U-Net.
*   **Mục đích:** Loại bỏ hoàn toàn mô hình Faster R-CNN nặng nề để tiết kiệm tài nguyên tính toán, tăng gấp đôi tốc độ xử lý (FPS) cho pipeline mà vẫn tự vẽ được bounding box chuẩn xác.
*   **Theo dõi (Object Tracking):** Sử dụng thuật toán **IoU Tracker** kết hợp bộ lọc khoảng cách để theo dõi (track) và gán ID duy nhất cho từng phương tiện qua các khung hình liên tục.

## 4. Ước lượng chuyển động Camera (Camera Ego-Motion Estimation)
*   **Luồng quang học (Optical Flow):** Sử dụng thuật toán luồng quang học Lucas-Kanade (`cv2.calcOpticalFlowPyrLK`) trên một lưới 32 điểm đặc trưng ở góc cao màn hình (vùng nền tĩnh) để tự động ước lượng tốc độ di chuyển của camera.
*   **Phân loại:** Phân biệt chính xác giữa Camera hành trình đang chạy (Dashcam) và Camera giám sát cố định (CCTV).

## 5. Công nghệ triển khai hệ thống
*   **Ngôn ngữ lập trình:** Python 3.8+
*   **Thư viện xử lý ảnh:** OpenCV (cv2) để đọc/ghi video, vẽ HUD, luồng quang học và biến đổi hình ảnh.
*   **Deep Learning Framework:** PyTorch để chạy suy luận mô hình phân đoạn U-Net ResNet50.
*   **Môi trường suy luận:** TensorFlow Lite cho mô hình độ sâu MiDaS.
