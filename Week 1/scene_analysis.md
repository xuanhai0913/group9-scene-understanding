# Logic Phân Tích Ngữ Cảnh Giao Thông

Đầu vào:
- Original Image

Các bước xử lý:
1. Xác định các đối tượng xuất hiện trong ảnh từ kết quả Semantic Segmentation.
2. Kết hợp thông tin từ Depth Map để ước lượng độ sâu của từng đối tượng.
3. Tính toán độ sâu trung bình theo từng lớp đối tượng.
4. Phân loại đối tượng theo khoảng cách:
   - Gần (Near)
   - Trung bình (Medium)
   - Xa (Far)
5. Tạo nhận xét và mô tả ngữ cảnh giao thông dựa trên các đối tượng và khoảng cách đã xác định.

Đầu ra:
- Hình ảnh Segmentation Mask, Segmentation Overlay, Depth Map.
- Hình ảnh Fusion kết hợp Segmentation và Depth.
