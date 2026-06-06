# User's Requirement

## Yêu cầu bài toán

Nhóm 9 chọn đề tài **phân tích ngữ cảnh giao thông dựa trên Semantic Segmentation và Depth Estimation**. Hệ thống nhận một ảnh đường phố làm đầu vào, sau đó tạo ra hai kết quả chính: ảnh phân vùng ngữ nghĩa và bản đồ độ sâu.

Semantic segmentation trả lời câu hỏi: **pixel này thuộc lớp nào?** Ví dụ, pixel có thể thuộc các lớp như mặt đường, xe, bầu trời, vỉa hè, người đi bộ, tòa nhà hoặc cây xanh.

Depth estimation trả lời câu hỏi: **vùng này gần hay xa camera?** Kết quả là depth map, trong đó mỗi điểm ảnh thể hiện mức độ gần xa tương đối.

## Input

Input dự kiến:

- Ảnh đường phố chụp từ góc nhìn xe hoặc camera giao thông.
- Ảnh có các thành phần quen thuộc như mặt đường, xe, bầu trời, vỉa hè, người đi bộ, nhà cửa, cây xanh.
- Ảnh có thể lấy từ Cityscapes, KITTI hoặc ảnh đường phố dùng cho demo.

## Output

Output mong muốn:

- Semantic segmentation mask.
- Ảnh segmentation overlay lên ảnh gốc.
- Depth map thể hiện vùng gần/xa.
- Ảnh tổng hợp gồm ảnh gốc, segmentation, depth và nhận xét ngắn.

## Scope

Phạm vi cần làm:

- Làm pipeline xử lý ảnh đường phố.
- Demo được segmentation và depth trên một hoặc nhiều ảnh.
- Giải thích được model và output.
- Có tài liệu trình bày theo luồng: requirement, features, tech solution, logic + AI, implement, test.
- Có thể dùng model pretrained nếu việc train quá nặng.

## Out Of Scope

Những phần không làm trong giai đoạn đầu:

- Không xây dựng ADAS hoàn chỉnh.
- Không điều khiển xe hoặc đưa ra quyết định an toàn thật.
- Không bắt buộc đo khoảng cách chính xác bằng mét.
- Không bắt buộc train toàn bộ Cityscapes hoặc KITTI.
- Không xử lý video real-time nếu thời gian không đủ.

## Tiêu chí thành công

Một demo được xem là đạt yêu cầu nếu:

- Ảnh đầu vào được đọc và hiển thị đúng.
- Segmentation mask thể hiện được các vùng chính như road, car, sky.
- Depth map cho thấy sự khác biệt gần/xa tương đối.
- Output được lưu lại rõ ràng.
- Nhóm giải thích được vì sao kết hợp segmentation và depth giúp hiểu cảnh tốt hơn.

## Câu tóm tắt khi thuyết trình

Đề tài của nhóm em nhận ảnh đường phố làm đầu vào, sau đó phân vùng từng pixel theo đối tượng và ước lượng độ sâu tương đối của từng vùng. Nhờ vậy hệ thống không chỉ biết trong ảnh có gì, mà còn hiểu được cấu trúc không gian gần xa trong cảnh giao thông.

