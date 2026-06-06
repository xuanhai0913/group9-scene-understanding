# Features

## Tổng quan tính năng

Hệ thống tập trung vào việc tạo kết quả trực quan từ ảnh đường phố. Các tính năng được chia thành ba nhóm: xử lý ảnh đầu vào, dự đoán bằng model và kết hợp kết quả để phân tích ngữ cảnh.

## Feature 1: Đọc và tiền xử lý ảnh

Mục tiêu:

- Đọc ảnh đường phố từ thư mục dữ liệu.
- Đổi ảnh về đúng định dạng màu.
- Resize ảnh về kích thước phù hợp với model.
- Chuẩn hóa ảnh trước khi inference.

Output:

- Ảnh gốc đã đọc thành công.
- Shape ảnh để kiểm tra.
- Ảnh sau tiền xử lý.

## Feature 2: Semantic Segmentation

Mục tiêu:

- Phân loại từng pixel trong ảnh.
- Tạo segmentation mask.
- Gán màu cho từng lớp.
- Tạo ảnh overlay giữa mask và ảnh gốc.

Các lớp ưu tiên:

- `road`
- `sidewalk`
- `car`
- `person`
- `sky`
- `building`
- `vegetation`

Output:

- `segmentation_mask.png`
- `segmentation_overlay.png`

## Feature 3: Depth Estimation

Mục tiêu:

- Ước lượng độ sâu từ một ảnh đơn.
- Tạo depth map thể hiện vùng gần/xa.
- Chuẩn hóa depth map để dễ hiển thị.

Output:

- `depth_map.png`
- Depth map dạng grayscale hoặc colormap.

Lưu ý:

- Với MiDaS, depth thường là độ sâu tương đối.
- Không nên nói đây là khoảng cách chính xác theo mét nếu chưa có calibration.

## Feature 4: Fusion Segmentation + Depth

Mục tiêu:

- Kết hợp semantic mask và depth map.
- Nhận xét ngữ cảnh giao thông.
- Làm nổi bật các vùng quan trọng như xe gần camera hoặc mặt đường phía trước.

Ví dụ nhận xét:

- Vùng `road` chiếm phần lớn phía dưới ảnh.
- Vùng `sky` thường nằm xa và ở phía trên ảnh.
- Vùng `car` có thể được so sánh gần/xa dựa trên depth map.
- Nếu xe nằm gần camera, hệ thống có thể đánh dấu là vùng cần chú ý.

## Feature 5: Visualization

Mục tiêu:

- Tạo hình tổng hợp để trình bày.
- Giúp giáo viên nhìn vào là hiểu pipeline.

Layout đề xuất:

```text
Original Image | Segmentation Mask | Segmentation Overlay | Depth Map | Fusion Result
```

## Feature 6: Report và AI Workflow

Mục tiêu:

- Có tài liệu giải thích rõ cách nhóm dùng AI.
- Có knowledge base trước khi hỏi AI.
- Có checklist test.
- Có phần edge cases và giới hạn dự án.

## Features List đầy đủ

- Đọc ảnh đầu vào.
- Kiểm tra ảnh đọc có hợp lệ không.
- Resize ảnh.
- Normalize ảnh.
- Chạy segmentation model.
- Sinh segmentation mask.
- Gán màu mask.
- Overlay mask lên ảnh gốc.
- Chạy depth model.
- Sinh depth map.
- Normalize depth map.
- Hiển thị depth bằng colormap.
- Kết hợp segmentation với depth.
- Tính depth trung bình theo class nếu có thời gian.
- Lưu output.
- Tạo notebook demo.
- Tạo báo cáo/slide.
- Tạo checklist test.
- Chuẩn bị câu hỏi phản biện.

## Câu tóm tắt khi thuyết trình

Feature của nhóm em tập trung vào hai output chính: segmentation mask và depth map. Segmentation giúp biết từng vùng trong ảnh là gì, còn depth giúp biết vùng đó gần hay xa. Khi kết hợp lại, hệ thống có thể phân tích ngữ cảnh giao thông tốt hơn.

