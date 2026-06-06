# Logic + AI

## Logic xử lý bài toán

Logic chính của dự án là tách bài toán scene understanding thành hai câu hỏi:

- Trong ảnh có những vùng nào và mỗi vùng thuộc lớp gì?
- Các vùng đó gần hay xa camera?

Câu hỏi thứ nhất được giải bằng semantic segmentation. Model nhận ảnh đầu vào và dự đoán nhãn cho từng pixel. Kết quả là một mask có cùng kích thước với ảnh, trong đó mỗi pixel thuộc một lớp như road, car, sky, sidewalk hoặc person.

Câu hỏi thứ hai được giải bằng depth estimation. Model nhận ảnh đầu vào và tạo depth map. Depth map giúp nhận xét cấu trúc không gian của cảnh, vì vùng gần camera và vùng xa camera sẽ có giá trị khác nhau.

## Logic Semantic Segmentation

Ảnh đầu vào được resize và normalize. Sau đó model segmentation dự đoán xác suất của từng lớp tại mỗi pixel. Class có xác suất cao nhất sẽ trở thành nhãn của pixel đó.

Kết quả:

- Mask dạng số: mỗi pixel là một class id.
- Mask dạng màu: mỗi class có một màu riêng.
- Overlay: trộn mask màu với ảnh gốc để quan sát dễ hơn.

## Logic Depth Estimation

Ảnh đầu vào được đưa vào MiDaS. Model tạo ra một depth map. Depth map được resize và normalize để hiển thị.

Điểm cần nhớ:

- MiDaS thường tạo relative depth.
- Relative depth cho biết vùng nào gần hơn hoặc xa hơn.
- Không nên khẳng định khoảng cách chính xác theo mét nếu không có calibration.

## Logic Fusion

Sau khi có segmentation mask và depth map, hệ thống kết hợp hai kết quả theo pixel.

Ví dụ:

- Nếu pixel thuộc lớp `road`, ta có thể quan sát độ sâu của mặt đường.
- Nếu pixel thuộc lớp `car`, ta có thể so sánh xe nào gần hơn.
- Nếu vùng `sky` có depth xa, kết quả hợp lý với ngữ cảnh.
- Nếu vùng `sidewalk` bị nhầm với `road`, cần kiểm tra lại segmentation.

Fusion giúp hệ thống hiểu cảnh ở mức cao hơn. Segmentation cho biết **vùng đó là gì**, depth cho biết **vùng đó gần hay xa**.

## Cách sử dụng AI

Nhóm dùng AI như công cụ hỗ trợ phân tích, không dùng để làm thay toàn bộ dự án.

Quy trình dùng AI:

- Chuẩn bị knowledge base trước khi hỏi.
- Tự viết lại yêu cầu bài toán.
- Hỏi AI kiểm tra cách hiểu.
- Hỏi AI góp ý feature.
- Hỏi AI so sánh model và dataset.
- Hỏi AI giải thích logic segmentation, depth và fusion.
- Hỏi AI lập checklist implement và test.
- Nhóm tự chốt phạm vi, viết code, chạy demo và kiểm tra output.

## Knowledge Base cho AI

Knowledge base nên đưa cho AI:

```text
Nhóm 9 làm đề tài scene understanding cho ảnh đường phố.
Input: ảnh đường phố.
Output: semantic segmentation mask và depth map.
Dataset tham khảo: Cityscapes cho segmentation, KITTI cho depth.
Model tham khảo: U-Net cho segmentation, MiDaS cho depth estimation.
Mục tiêu: demo phân tích ngữ cảnh giao thông, không xây dựng xe tự hành hoàn chỉnh.
AI chỉ hỗ trợ giải thích, góp ý logic, chia task và lập checklist test.
```

## Prompt mẫu

```text
Nhóm em đang làm đề tài phân tích ngữ cảnh giao thông bằng semantic segmentation và depth estimation. Em đã tự xác định input là ảnh đường phố, output là segmentation mask và depth map. Bạn kiểm tra giúp cách hiểu này đã đúng chưa, cần giới hạn phạm vi thế nào để dễ triển khai trong bài tập lớn.
```

```text
Với đề tài này, em dự kiến dùng Cityscapes cho segmentation, KITTI cho depth, U-Net cho segmentation và MiDaS cho depth estimation. Bạn phân tích giúp vì sao cách chọn này hợp lý, điểm mạnh và hạn chế của từng thành phần.
```

```text
Em muốn giải thích logic kết hợp segmentation và depth. Bạn giúp em diễn đạt ngắn gọn: segmentation trả lời pixel thuộc lớp gì, depth trả lời vùng đó gần hay xa, khi kết hợp thì hệ thống hiểu ngữ cảnh giao thông tốt hơn.
```

## Câu tóm tắt khi thuyết trình

Nhóm em dùng AI như một công cụ hỗ trợ phân tích. Trước khi hỏi AI, nhóm em đưa knowledge base gồm yêu cầu, input, output, dataset, model và giới hạn đề tài. AI được dùng để kiểm tra logic, gợi ý cách triển khai và lập checklist, còn việc chốt phạm vi, viết code, chạy demo và đánh giá kết quả là do nhóm tự thực hiện.

