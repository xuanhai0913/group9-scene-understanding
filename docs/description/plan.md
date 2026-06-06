# Plan

## Mục tiêu

Hoàn thành bài tập lớn của Nhóm 9 với đề tài **phân tích ngữ cảnh giao thông bằng semantic segmentation và depth estimation**. Dự án cần có tài liệu rõ ràng, demo chạy được, output trực quan và log quá trình làm việc với Agent AI.

## Giai đoạn 1: Chốt phạm vi và tài liệu

Việc cần làm:

- Chốt tên đề tài.
- Chốt input/output.
- Viết project description.
- Viết user's requirement.
- Viết features.
- Viết tech solutions.
- Viết logic + AI.
- Viết implement plan.
- Viết test plan.
- Tạo log discussion cho quá trình dùng AI.

Kết quả mong muốn:

- Repo có thư mục `docs/`.
- Repo có thư mục `docs/description/`.
- Tài liệu có tiếng Việt có dấu.
- Có mô tả rõ scope, out of scope, edge cases, features list và tech solution.

## Giai đoạn 2: Chuẩn bị kỹ thuật

Việc cần làm:

- Tạo môi trường Python.
- Cài OpenCV, NumPy, Matplotlib, PyTorch.
- Chọn ảnh mẫu đường phố.
- Tìm model segmentation phù hợp.
- Tìm model MiDaS hoặc Depth Anything nếu cần thay thế.
- Tạo notebook demo.

Kết quả mong muốn:

- Notebook đọc được ảnh.
- Hiển thị được ảnh gốc.
- Có pipeline code rõ ràng.

## Giai đoạn 3: Semantic Segmentation

Việc cần làm:

- Load segmentation model.
- Chạy inference trên ảnh đường phố.
- Tạo segmentation mask.
- Gán màu cho các lớp.
- Overlay mask lên ảnh gốc.

Kết quả mong muốn:

- Có ảnh `segmentation_mask.png`.
- Có ảnh `segmentation_overlay.png`.
- Giải thích được segmentation là phân loại từng pixel.

## Giai đoạn 4: Depth Estimation

Việc cần làm:

- Load MiDaS.
- Chạy inference trên cùng ảnh đầu vào.
- Tạo depth map.
- Normalize depth map để hiển thị.
- Lưu depth output.

Kết quả mong muốn:

- Có ảnh `depth_map.png`.
- Giải thích được depth là gần/xa tương đối.
- Không nói quá thành khoảng cách mét chính xác nếu chưa có calibration.

## Giai đoạn 5: Fusion và phân tích ngữ cảnh

Việc cần làm:

- Căn chỉnh segmentation mask và depth map.
- Kết hợp hai output.
- Nhận xét theo từng vùng.
- Tạo ảnh tổng hợp.

Ví dụ nhận xét:

```text
Road nằm phía dưới ảnh và có xu hướng xa dần về phía chân trời.
Car nằm bên trái ảnh và có depth gần hơn nền.
Sky là vùng phía trên và thường được hiểu là vùng xa.
```

Kết quả mong muốn:

- Có ảnh `fusion_result.png`.
- Có nhận xét ngữ cảnh giao thông ngắn.

## Giai đoạn 6: Test

Việc cần làm:

- Test preprocessing.
- Test segmentation.
- Test depth.
- Test fusion.
- Test edge cases.
- Ghi lại lỗi và cách xử lý.

Checklist:

- Ảnh đọc được.
- Shape đúng.
- Không sai màu RGB/BGR.
- Mask đúng kích thước.
- Depth map không rỗng.
- Overlay nhìn rõ.
- Output lưu đúng thư mục.
- Nhận xét không overclaim.

## Giai đoạn 7: Báo cáo và trình bày

Việc cần làm:

- Làm slide.
- Làm báo cáo.
- Chuẩn bị câu hỏi phản biện.
- Chuẩn bị giải thích AI workflow.
- Chuẩn bị demo notebook/script.

Phân công gợi ý:

- Hải: leader, scope, AI workflow, đối đáp.
- Thành viên 1: dataset.
- Thành viên 2: segmentation.
- Thành viên 3: depth estimation.
- Thành viên 4: fusion + visualization.
- Thành viên 5: report + slide + test checklist.

## Rủi ro và phương án dự phòng

Nếu train segmentation quá nặng:

- Dùng pretrained model.
- Chạy demo inference thay vì train từ đầu.

Nếu MiDaS chạy chậm:

- Giảm kích thước ảnh.
- Chạy ít ảnh demo hơn.

Nếu dataset khó tải:

- Dùng ảnh mẫu từ KITTI/Cityscapes hoặc ảnh đường phố hợp lệ để demo.

Nếu depth bị hỏi về khoảng cách:

- Trả lời rõ đây là relative depth, không phải khoảng cách mét tuyệt đối.

