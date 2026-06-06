# Test

## Mục tiêu test

Test dùng để chứng minh pipeline chạy đúng và output có ý nghĩa. Vì dự án có hai nhánh là segmentation và depth, test cần kiểm tra riêng từng nhánh, sau đó kiểm tra phần kết hợp.

## Test Input

Cần chuẩn bị nhiều ảnh đường phố:

- Ảnh có mặt đường rõ ràng.
- Ảnh có xe gần và xe xa.
- Ảnh có bầu trời và vỉa hè.
- Ảnh có nhiều đối tượng nhỏ.
- Ảnh thiếu sáng hoặc có vật cản nếu muốn kiểm tra thêm edge cases.

## Test Preprocessing

Cần kiểm tra:

- Ảnh đọc được, không bị `None`.
- Shape ảnh đúng.
- Ảnh không bị sai màu BGR/RGB.
- Resize không làm méo tỷ lệ nếu yêu cầu giữ tỷ lệ.
- Normalize đúng format input của model.

Nếu fail:

- Kiểm tra đường dẫn file.
- Kiểm tra định dạng ảnh.
- Kiểm tra thứ tự kênh màu.
- Kiểm tra kích thước đầu vào của model.

## Test Segmentation

Cần kiểm tra:

- Mask có cùng kích thước với ảnh hoặc được resize về đúng kích thước.
- Mỗi pixel trong mask có class hợp lệ.
- Các lớp chính như road, sky, car xuất hiện hợp lý.
- Overlay không che hoàn toàn ảnh gốc.

Metric nếu có ground truth:

- Pixel accuracy.
- Mean IoU.
- Per-class IoU cho road, car, sky.

Nếu không có ground truth:

- Đánh giá trực quan trên ảnh demo.
- So sánh với ảnh gốc và ghi nhận lỗi rõ ràng.

## Test Depth

Cần kiểm tra:

- Depth map tạo ra được, không bị rỗng.
- Depth map được resize về đúng kích thước ảnh gốc.
- Vùng gần và xa có sự khác biệt rõ.
- Colormap hiển thị dễ quan sát.

Điều cần nói rõ:

- Depth của MiDaS là gần xa tương đối.
- Không nên khẳng định khoảng cách mét nếu không có calibration hoặc ground truth.

Metric nếu có ground truth depth:

- MAE.
- RMSE.
- Abs Rel.

Nếu không có ground truth:

- Đánh giá tương quan trực quan: đường phía xa nhỏ dần, bầu trời/nền xa, xe gần nổi bật hơn.

## Test Fusion

Cần kiểm tra:

- Segmentation mask và depth map được căn chỉnh cùng kích thước.
- Khi lấy depth theo class, không bị sai index pixel.
- Nhận xét ngữ cảnh phù hợp với ảnh.

Ví dụ test:

```text
Nếu vùng car nằm gần camera, depth của vùng car phải thể hiện mức gần hơn so với nền xa.
Nếu vùng sky nằm phía trên ảnh, depth thường thể hiện là vùng xa.
Nếu road nằm phía dưới ảnh, depth thay đổi theo hướng xa dần về phía chân trời.
```

## Test Edge Cases

Cần thử thêm:

- Ảnh ban đêm.
- Ảnh có mưa hoặc sương mù.
- Ảnh bị lóa sáng.
- Đường bị che khuất bởi xe lớn.
- Xe quá nhỏ hoặc ở quá xa.
- Người đi bộ bị che một phần.
- Vạch đường mờ.
- Cảnh không giống dataset huấn luyện.

## Checklist trước khi nộp

- Có ít nhất 3 ảnh demo.
- Mỗi ảnh có ảnh gốc, segmentation, depth và fusion.
- Có giải thích model dùng gì và output là gì.
- Có nói rõ hạn chế relative depth.
- Có file notebook hoặc script chạy được.
- Có thư mục outputs lưu kết quả.
- Có slide hoặc báo cáo tóm tắt pipeline.
- Có phần AI workflow và knowledge base.
- Có phần edge cases.
- Có phần scope và out of scope.

## Câu tóm tắt khi thuyết trình

Nhóm em test theo từng phần: đầu tiên kiểm tra ảnh đầu vào và tiền xử lý, sau đó kiểm tra segmentation mask, depth map, rồi mới kiểm tra phần kết hợp. Nếu có ground truth thì dùng metric như IoU cho segmentation và RMSE cho depth. Nếu không có ground truth, nhóm em đánh giá trực quan trên nhiều ảnh và nêu rõ hạn chế của mô hình.

