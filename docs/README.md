# Group 9 Scene Understanding

## Tên đề tài

**Phân tích ngữ cảnh giao thông dựa trên Semantic Segmentation và Depth Estimation**

Tên tiếng Anh đề xuất:

```text
Traffic Scene Understanding using Semantic Segmentation and Depth Estimation
```

## Mô tả dự án

Dự án xử lý ảnh đường phố để giúp hệ thống hiểu ngữ cảnh giao thông ở mức trực quan. Với một ảnh đầu vào, hệ thống tạo ra hai loại kết quả chính:

- **Semantic segmentation**: phân vùng từng pixel theo lớp ngữ nghĩa như `road`, `car`, `sky`, `sidewalk`, `person`, `building`.
- **Depth estimation**: tạo bản đồ độ sâu để ước lượng vùng nào gần camera và vùng nào xa camera.

Khi kết hợp hai kết quả này, hệ thống không chỉ biết trong ảnh có những đối tượng nào, mà còn hiểu được cấu trúc không gian của cảnh. Ví dụ: đâu là mặt đường phía trước, đâu là xe, xe nào nằm gần camera hơn, vùng nào là bầu trời hoặc nền xa.

## Scope

Phạm vi chính của dự án:

- Nhận ảnh đường phố làm input.
- Tiền xử lý ảnh: đọc ảnh, resize, chuẩn hóa, đổi định dạng màu nếu cần.
- Chạy semantic segmentation để tạo mask theo từng lớp.
- Chạy depth estimation để tạo depth map.
- Hiển thị kết quả segmentation, depth và overlay.
- Kết hợp segmentation với depth để đưa ra nhận xét về ngữ cảnh giao thông.
- Tạo notebook hoặc script demo chạy được trên một số ảnh mẫu.
- Viết báo cáo giải thích pipeline, model, dataset, kết quả và hạn chế.

## Out Of Scope

Những phần không nằm trong phạm vi giai đoạn đầu:

- Không xây dựng hệ thống xe tự hành hoàn chỉnh.
- Không điều khiển xe, phanh, lái hoặc ra quyết định thật.
- Không yêu cầu ước lượng khoảng cách tuyệt đối theo mét nếu không có camera calibration hoặc ground truth depth.
- Không bắt buộc train model lớn từ đầu trên toàn bộ Cityscapes/KITTI.
- Không xử lý real-time video trong phiên bản đầu.
- Không nhận dạng biển báo chi tiết hoặc tracking đối tượng theo thời gian.
- Không đánh giá an toàn giao thông ở mức triển khai thực tế.

## Edge Cases

Các trường hợp cần lưu ý khi kiểm thử:

- Ảnh ban đêm, thiếu sáng hoặc bị lóa đèn.
- Ảnh mưa, sương mù, bụi hoặc thời tiết xấu.
- Xe bị che khuất một phần bởi vật thể khác.
- Đường có bóng cây, vạch kẻ mờ hoặc mặt đường phản chiếu.
- Ảnh có góc nhìn khác dữ liệu train, ví dụ camera quá cao hoặc quá thấp.
- Ảnh có nhiều người, nhiều xe, nhiều vật thể nhỏ.
- Bầu trời bị che bởi nhà cao tầng hoặc cây.
- Depth map có thể sai ở vùng gương, kính, vật thể phản xạ hoặc vật thể quá mỏng.
- Segmentation có thể nhầm giữa road và sidewalk nếu biên không rõ.

## Features List

Các tính năng chính:

- Đọc ảnh đường phố từ thư mục dữ liệu.
- Tiền xử lý ảnh đầu vào.
- Hiển thị ảnh gốc.
- Dự đoán semantic segmentation mask.
- Gán màu cho từng lớp segmentation.
- Overlay segmentation mask lên ảnh gốc.
- Dự đoán depth map từ ảnh đơn.
- Chuẩn hóa depth map để hiển thị.
- Hiển thị depth map bằng colormap.
- Kết hợp segmentation và depth.
- Tính hoặc nhận xét độ sâu trung bình theo từng lớp.
- Tạo hình tổng hợp gồm ảnh gốc, segmentation, depth và fusion.
- Lưu kết quả vào thư mục `outputs`.
- Tạo checklist test cho từng module.
- Chuẩn bị phần giải thích AI workflow và knowledge base.

## Tech Solution

Giải pháp kỹ thuật đề xuất:

- **Ngôn ngữ**: Python.
- **Xử lý ảnh cơ bản**: OpenCV, NumPy.
- **Hiển thị kết quả**: Matplotlib.
- **Deep learning**: PyTorch.
- **Semantic segmentation**: U-Net hoặc model segmentation pretrained.
- **Depth estimation**: MiDaS pretrained model.
- **Dataset tham khảo**:
  - Cityscapes cho semantic segmentation.
  - KITTI cho ảnh giao thông và depth.
- **Output demo**:
  - `original_image.png`
  - `segmentation_mask.png`
  - `segmentation_overlay.png`
  - `depth_map.png`
  - `fusion_result.png`

## Thứ tự tài liệu

Tài liệu được viết theo đúng thứ tự trình bày:

1. [User's Requirement](./01_users_requirement.md)
2. [Features](./02_features.md)
3. [Tech Solutions](./03_tech_solutions.md)
4. [Logic + AI](./04_logic_ai.md)
5. [Implement](./05_implement.md)
6. [Test](./06_test.md)

## Tên repo

Tên repo chính:

```text
group9-scene-understanding
```

Một số tên thay thế:

```text
group9-traffic-segmentation-depth
cvip-group9-scene-understanding
semantic-depth-road-scene
traffic-scene-understanding-group9
```

