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

## 6 role để thành viên chọn trước

Trước khi bắt đầu code, nhóm nên để 6 thành viên chọn role trước. Mỗi role có đầu việc riêng, nhưng khi làm báo cáo/demo vẫn cần phối hợp với nhau.

### Role 1: Leader + AI Workflow + Tổng quan

Phụ trách:

- Chốt scope và out of scope.
- Điều phối tiến độ nhóm.
- Chuẩn bị knowledge base cho AI.
- Ghi hoặc duyệt log AI khi cần.
- Nắm pipeline tổng thể để trả lời đối đáp.
- Kết nối các phần dataset, segmentation, depth, fusion, test.

Output cần có:

- Phần giới thiệu đề tài.
- AI workflow.
- Câu trả lời tổng quan khi giáo viên hỏi.

### Role 2: Dataset + Requirement

Phụ trách:

- Tìm hiểu Cityscapes và KITTI.
- Giải thích vì sao chọn dataset.
- Chọn ảnh mẫu để demo.
- Mô tả input/output.
- Nêu các giới hạn của dữ liệu.

Output cần có:

- File/slide mô tả dataset.
- Danh sách ảnh mẫu.
- Giải thích input/output.

### Role 3: Semantic Segmentation

Phụ trách:

- Tìm hiểu semantic segmentation.
- Tìm hiểu U-Net hoặc model segmentation pretrained.
- Chạy hoặc chuẩn bị phần segmentation.
- Tạo segmentation mask.
- Tạo overlay mask lên ảnh gốc.

Output cần có:

- Ảnh segmentation mask.
- Ảnh segmentation overlay.
- Giải thích segmentation là phân loại từng pixel.

### Role 4: Depth Estimation

Phụ trách:

- Tìm hiểu depth estimation.
- Tìm hiểu MiDaS.
- Chạy hoặc chuẩn bị phần depth map.
- Giải thích relative depth.
- Nêu hạn chế khi không có khoảng cách mét tuyệt đối.

Output cần có:

- Depth map.
- Giải thích vùng gần/xa.
- Câu trả lời khi giáo viên hỏi về độ sâu.

### Role 5: Fusion + Visualization

Phụ trách:

- Kết hợp segmentation mask và depth map.
- Tạo ảnh tổng hợp.
- Viết nhận xét ngữ cảnh giao thông.
- Làm output dễ nhìn để demo.
- Kiểm tra các hình kết quả có rõ ràng không.

Output cần có:

- Fusion result.
- Hình tổng hợp: original, segmentation, depth, fusion.
- Nhận xét ngắn cho từng ảnh demo.

### Role 6: Test + Report + Slide

Phụ trách:

- Lập checklist test.
- Kiểm tra preprocessing, segmentation, depth và fusion.
- Ghi lỗi thường gặp.
- Chuẩn bị báo cáo và slide.
- Chuẩn bị câu hỏi phản biện.

Output cần có:

- Test checklist.
- Danh sách edge cases.
- Slide/báo cáo tổng hợp.
- Câu hỏi và câu trả lời dự phòng.

## Phân công gợi ý nếu cần chốt nhanh

Nếu nhóm muốn chia nhanh, có thể dùng mẫu sau:

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
