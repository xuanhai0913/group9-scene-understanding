# Task Roles

File này dùng để nhóm chọn role trước khi triển khai dự án. Nhóm có 6 thành viên, mỗi người nên chọn một role chính để tránh trùng việc và dễ chịu trách nhiệm khi thuyết trình.

## Role 1: Leader + AI Workflow + Tổng quan

Phù hợp với người:

- Nắm được toàn bộ đề tài.
- Có thể điều phối nhóm.
- Có thể trả lời câu hỏi tổng quan.

Nhiệm vụ:

- Chốt scope và out of scope.
- Chuẩn bị knowledge base cho AI.
- Kiểm soát log AI, chỉ ghi khi được phép.
- Nắm pipeline tổng thể.
- Hỗ trợ các thành viên khi bị hỏi lan man.

Phần cần trình bày:

- Vì sao chọn đề tài.
- Input/output của hệ thống.
- AI được dùng như thế nào.
- Giới hạn của dự án.

## Role 2: Dataset + Requirement

Phù hợp với người:

- Chịu đọc tài liệu dataset.
- Biết giải thích dữ liệu đầu vào/đầu ra.

Nhiệm vụ:

- Tìm hiểu Cityscapes.
- Tìm hiểu KITTI.
- Chọn ảnh mẫu đường phố.
- Mô tả dữ liệu dùng cho segmentation và depth.
- Nêu edge cases liên quan đến dữ liệu.

Phần cần trình bày:

- Cityscapes dùng cho gì.
- KITTI dùng cho gì.
- Vì sao dataset phù hợp với giao thông.
- Hạn chế nếu dữ liệu khác môi trường thật.

## Role 3: Semantic Segmentation

Phù hợp với người:

- Muốn làm phần model phân vùng ảnh.
- Có thể giải thích pixel/class/mask.

Nhiệm vụ:

- Tìm hiểu semantic segmentation.
- Tìm hiểu U-Net hoặc pretrained segmentation model.
- Tạo segmentation mask.
- Gán màu cho từng class.
- Tạo segmentation overlay.

Phần cần trình bày:

- Semantic segmentation là gì.
- Khác gì với object detection.
- U-Net hoạt động tổng quan thế nào.
- Mask và overlay được tạo ra sao.

## Role 4: Depth Estimation

Phù hợp với người:

- Muốn làm phần độ sâu/gần xa.
- Có thể giải thích MiDaS và relative depth.

Nhiệm vụ:

- Tìm hiểu depth estimation.
- Tìm hiểu MiDaS.
- Chạy hoặc chuẩn bị depth map.
- Normalize depth để hiển thị.
- Nêu hạn chế của relative depth.

Phần cần trình bày:

- Depth estimation là gì.
- MiDaS nhận input gì, output gì.
- Vì sao depth ở đây là gần/xa tương đối.
- Vì sao không khẳng định khoảng cách mét tuyệt đối.

## Role 5: Fusion + Visualization

Phù hợp với người:

- Có mắt nhìn output.
- Muốn làm phần kết hợp kết quả và demo.

Nhiệm vụ:

- Kết hợp segmentation và depth.
- Tạo ảnh tổng hợp.
- Viết nhận xét ngữ cảnh giao thông.
- Làm output dễ nhìn.
- Chuẩn bị hình demo cho slide.

Phần cần trình bày:

- Vì sao cần kết hợp segmentation và depth.
- Cách đọc fusion result.
- Ví dụ: xe nào gần hơn, road nằm ở đâu, sky là vùng xa.

## Role 6: Test + Report + Slide

Phù hợp với người:

- Cẩn thận, thích rà lỗi và làm tài liệu.
- Có thể chuẩn bị slide/báo cáo.

Nhiệm vụ:

- Viết checklist test.
- Test preprocessing.
- Test segmentation.
- Test depth.
- Test fusion.
- Chuẩn bị báo cáo và slide.
- Tổng hợp câu hỏi phản biện.

Phần cần trình bày:

- Nhóm đã test những gì.
- Lỗi dễ gặp là gì.
- Edge cases nào cần lưu ý.
- Hạn chế của demo.

## Mẫu tin nhắn gửi nhóm

```text
Mọi người chọn role trước nha, nhóm mình có 6 role:

1. Leader + AI Workflow + Tổng quan
2. Dataset + Requirement
3. Semantic Segmentation
4. Depth Estimation
5. Fusion + Visualization
6. Test + Report + Slide

Mỗi người chọn 1 role chính. Khi chọn xong thì đọc kỹ phần mình phụ trách, chuẩn bị giải thích được input, output, model/logic và lỗi dễ gặp. Mục tiêu là ai làm phần nào thì khi cô hỏi phần đó phải trả lời được.
```

