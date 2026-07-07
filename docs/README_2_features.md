# README 2: Features (Các tính năng của hệ thống)

Tài liệu này mô tả các tính năng chính được triển khai trong dự án ADAS và hiểu ngữ cảnh giao thông.

---

## 1. Phân đoạn làn đường & Phân biệt các thành phần trên đường (Semantic Road Scene Segmentation)
Hệ thống đã triển khai phân đoạn ngữ cảnh toàn diện và phân biệt rõ ràng các nhóm thực thể trên đường (dựa trên bộ dữ liệu chuẩn Cityscapes) với mã màu hiển thị trực quan chi tiết như sau:

*   **Làn đường có thể đi (Drivable Lane):**
    *   **Vùng mặt đường (Road/Flat):** Được mô hình phân đoạn và tô **Màu Tím Bán Trong Suốt** (Purple - BGR: `(128, 64, 128)`) trên màn hình mặt nạ phân đoạn.
    *   **Lớp phủ làn đường đang di chuyển (Ego-lane Overlay):** Được thuật toán lọc thông minh xác định biên và phủ **Màu Xanh Lá Cây** (Green - BGR: `(0, 255, 0)`) trực tiếp lên luồng video chính để biểu diễn vùng di chuyển an toàn của xe chủ.
*   **Chướng ngại vật trên đường (Obstacles):**
    *   **Phương tiện giao thông (Ô tô, xe máy, xe tải, xe bus):** Được phân đoạn bằng **Màu Đỏ** (Red - BGR: `(0, 0, 255)`) trên mặt nạ. Trên màn hình HUD chính, phương tiện được khoanh vùng bằng khung hình chữ nhật kèm chỉ số khoảng cách tương đối (ví dụ: `CAR #1: 4.2 rel`). Màu sắc khung bao thay đổi linh hoạt theo mức độ nguy hiểm:
        *   *Màu Xanh Lá Cây* (An toàn, khoảng cách xa).
        *   *Màu Cam* (Cảnh báo xe tạt đầu / cutting-in).
        *   *Màu Đỏ* (chỉ số gần thỏa điều kiện cảnh báo heuristic của demo).
    *   **Người đi bộ (Human/Pedestrians):** Được phân đoạn bằng **Màu Hồng Đỏ** (Pink-Red - BGR: `(60, 20, 220)`) trên mặt nạ để nhận diện nhanh chóng.
*   **Môi trường xung quanh (Surroundings):**
    *   **Vỉa hè / Công trình (Sidewalk/Construction):** Được phân đoạn bằng **Màu Xám** (Gray - BGR: `(70, 70, 70)`).
    *   **Cây cối / Tự nhiên (Trees/Nature):** Được phân đoạn bằng **Màu Xanh Lá Cây Đậm** (Dark Green - BGR: `(35, 142, 107)`).
    *   **Bầu trời (Sky):** Được phân đoạn bằng **Màu Xanh Lam** (Sky Blue - BGR: `(180, 130, 70)`).
    *   **Biển báo / Cột mốc / Đèn tín hiệu (Traffic Lights/Signs/Objects):** Được phân đoạn bằng **Màu Xám Sáng** (Light Gray - BGR: `(153, 153, 153)`).

## 2. Ước lượng độ sâu dạng Dashboard (Real-time Depth Mapping)
*   Hệ thống ước lượng khoảng cách cho toàn bộ điểm ảnh trong khung hình sử dụng mô hình **MiDaS TFLite** và hiển thị một màn hình phụ (Dashboard) bản đồ nhiệt độ sâu.
*   Các vùng ở gần xe có màu ấm (đỏ/cam/vàng) thể hiện khoảng cách nguy hiểm, các vùng ở xa có màu lạnh (xanh dương/tím).

## 3. Trích xuất hộp bao chướng ngại vật động (Contours-based Bounding Box Extraction)
*   Thay vì chạy thêm mô hình Faster R-CNN nặng nề, hệ thống sử dụng giải pháp lai thông minh: **Trích xuất trực tiếp hộp bao (Bounding Box) từ mặt nạ phân đoạn lớp Phương tiện (Vehicle - Màu đỏ) và Con người (Human - Màu hồng) của U-Net** bằng thuật toán tìm đường bao OpenCV (`cv2.findContours`).
*   Gán hộp bao (Bounding Box) kèm chỉ số khoảng cách tương đối được cập nhật động từng khung hình (ví dụ: `VEHICLE #32: 9.4 rel`).

## 4. Tự động nhận diện chuyển động Camera bằng Luồng quang học (Optical Flow)
*   Tự động tính toán luồng quang học Lucas-Kanade (`cv2.calcOpticalFlowPyrLK`) cho các điểm nền tĩnh ở góc cao khung hình để đo tốc độ dịch chuyển của camera trong 1 giây đầu.
*   **Camera Cố định (CCTV):** Tốc độ dịch chuyển nền bằng 0 $\rightarrow$ Tự động chuyển sang chế độ **Giám sát toàn diện mặt đường (Full Road)**.
*   **Camera Hành trình (Dashcam):** Tốc độ dịch chuyển nền lớn hơn ngưỡng $\rightarrow$ Tự động chuyển sang chế độ **Chia làn đường (Split Road)** để hỗ trợ lái xe an toàn.

## 5. Bộ định vị làn tự thích ứng thông minh (Self-Adaptive Lead-Vehicle Guided Lane Selector)
*   Khi chạy chế độ chia làn đường (`--split_road`), hệ thống chạy cơ chế bình chọn thông minh trong 45 khung hình đầu tiên dựa trên hướng của các xe dẫn đường (lead vehicle) đi cùng chiều phía trước (lọc bằng độ sâu và chuyển động tương đối `delta_d < 0.25` để loại bỏ xe ngược chiều và xe đỗ bên đường):
    *   **Làn bên trái (Left Ego):** Thích hợp cho xe máy đi làn trái (Video 1 TP. HCM) hoặc xe chạy ở quốc gia đi bên trái (Anh, Nhật). Hệ thống vẽ làn Ego ở bên trái và vẽ vạch đứt màu vàng đè lên dải phân cách cứng ở giữa.
    *   **Làn ở giữa (Center Ego):** Thích hợp cho ô tô đi giữa làn (Video 2 Dashcam). Hệ thống vẽ làn Ego ở giữa (dịch trái nhẹ 8% để khớp thực tế xe chạy) với cả hai biên màu xanh lá cây.
    *   **Làn bên phải (Right Ego):** Mặc định cho camera tĩnh nếu có xe đi lệch phải.
*   Sau 45 khung hình, hệ thống sẽ **khóa cứng (lock)** kiểu làn đường này để giữ giao diện hiển thị ổn định 100%, không bị nhấp nháy hoặc thay đổi đột ngột.

## 6. Cảnh báo va chạm tức thời (Instant Collision Alert)
*   Tính toán va chạm thông minh chỉ áp dụng cho phương tiện đi cùng làn di chuyển chính (Ego Lane - Màu Xanh Lá Cây). Phương tiện ở làn đối diện hoặc vỉa hè sẽ không bị cảnh báo sai lệch.
*   Khi chỉ số tương đối thỏa điều kiện cảnh báo heuristic (khoảng cách tương đối < 4.5), HUD hiển thị banner đỏ `COLLISION WARNING: Obstacle too close!`. Đây là minh họa nghiên cứu, không phải hệ thống cảnh báo an toàn đã hiệu chuẩn.
