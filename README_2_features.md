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
*   Hệ thống ước lượng khoảng cách cho toàn bộ điểm ảnh trong khung hình và hiển thị một màn hình phụ (Dashboard) bản đồ nhiệt độ sâu.
*   Các vùng ở gần xe có màu ấm (đỏ/cam/vàng) thể hiện khoảng cách nguy hiểm, các vùng ở xa có màu lạnh (xanh dương/tím).

## 3. Phát hiện và Ước lượng gần/xa phương tiện (Vehicle Detection & Relative Depth HUD)
*   Tự động phát hiện vị trí các phương tiện giao thông (xe hơi, xe máy, xe tải) di chuyển phía trước.
*   Gắn hộp bao (Bounding Box) kèm chỉ số khoảng cách tương đối được cập nhật động từng khung hình (ví dụ: `Car: 4.2 rel`).

## 4. Cảnh báo va chạm tức thời (Instant Collision Alert)
*   Tính toán va chạm thông minh chỉ áp dụng cho phương tiện đi cùng làn (phương tiện ở làn đối diện hoặc lề đường sẽ không bị cảnh báo sai lệch).
*   Khi chỉ số tương đối thỏa điều kiện cảnh báo heuristic, HUD hiển thị banner `WARNING: Front vehicle too close!`. Đây là minh họa nghiên cứu, không phải hệ thống cảnh báo an toàn đã hiệu chuẩn.
