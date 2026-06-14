# README 1: User Requirements (Yêu cầu hệ thống)

Tài liệu này mô tả chi tiết các yêu cầu nghiệp vụ và kỹ thuật của hệ thống hỗ trợ lái xe nâng cao (ADAS) kết hợp Phân đoạn ngữ cảnh (Semantic Segmentation) và Ước lượng độ sâu (Depth Estimation).

---

## 1. Bối cảnh và Bài toán
Trong các hệ thống tự hành và hỗ trợ lái xe nâng cao (ADAS), việc hiểu rõ môi trường xung quanh (Traffic Scene Understanding) là yếu tố sống còn để đảm bảo an toàn. 
Đề tài này tập trung giải quyết bài toán: **Segmentation + Depth -> Scene understanding** (Phân đoạn đường + Tính toán độ sâu -> Hiểu ngữ cảnh giao thông).

---

## 2. Các yêu cầu cốt lõi từ người dùng (User's Requirements)

### 2.1. Yêu cầu Phân đoạn ngữ cảnh (Semantic Segmentation)
*   **Mục tiêu:** Hệ thống phải phân tách được chính xác khu vực mặt đường (Road) và phân chia các làn đường trên video thời gian thực.
*   **Yêu cầu kỹ thuật:** Sử dụng dữ liệu thực tế từ tập dữ liệu **KITTI** và mô hình phân đoạn mức độ pixel để xác định ranh giới đường đi.

### 2.2. Yêu cầu Ước lượng độ sâu (Depth Estimation)
*   **Mục tiêu:** Tính toán bản đồ độ sâu của khung cảnh phía trước từ một camera đơn sắc (Monocular Camera) để biết khoảng cách xa/gần của các chướng ngại vật.
*   **Yêu cầu kỹ thuật:** Sử dụng mô hình ước lượng độ sâu thời gian thực để tạo ra bản đồ nhiệt độ sâu tương ứng với các khung hình video.

### 2.3. Yêu cầu Kết hợp thông tin & Cảnh báo an toàn (Collision Warning)
*   **Mục tiêu:** Phát hiện các phương tiện đi phía trước (ô tô, xe máy) và theo dõi khoảng cách thực tế của chúng đối với xe chủ.
*   **Quy tắc cảnh báo:** Nếu phương tiện di chuyển **nằm trong làn đường hiện tại** của xe chủ và khoảng cách **dưới 5 mét**, hệ thống phải ngay lập tức đưa ra cảnh báo nguy hiểm trực quan trên màn hình điều khiển (HUD).
