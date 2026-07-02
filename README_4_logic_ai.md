# README 4: Logic & AI Fusion (Logic kết hợp dữ liệu AI)

Tài liệu này giải thích thuật toán logic tích hợp đầu ra từ các mô hình AI khác nhau để đưa ra quyết định cảnh báo va chạm.

---

## 1. Sơ đồ xử lý tích hợp dữ liệu
Mỗi khung hình video đầu vào đi qua 3 luồng xử lý song song trước khi được tích hợp:

```
                  ┌───────────────┐
                  │  Video Frame  │
                  └───────┬───────┘
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │   U-Net   │    │   MiDaS   │    │  FR-CNN   │
   └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
         │ (Mask)         │ (Depth)        │ (BBoxes)
         ▼                │                ▼
 ┌───────────────┐        │        ┌───────────────┐
 │ Định vị làn xe│        │        │Ước lượng gần/xa│
 └───────┬───────┘        │        └───────┬───────┘
         │                ▼                │
         └─────────► TÍCH HỢP ◄────────────┘
                        │
                        ▼
            ┌──────────────────────┐
            │ Quyết định cảnh báo  │
            └──────────────────────┘
```

---

## 2. Các thuật toán logic cốt lõi

### 2.1. Thuật toán định vị vùng làn đường (Lane Area definition)
*   Sử dụng hàm hỗ trợ `detect_lanes` trong `utils/lane.py` để xác định 4 điểm neo ranh giới bao gồm: Cận trái dưới, cận trái trên, cận phải dưới, cận phải trên.
*   Thiết lập các đa giác làn đường tương ứng (Ego Lane Polygon Area) dựa trên các tọa độ biên này để khoanh vùng khu vực theo dõi vật cản của xe chủ.

### 2.2. Thuật toán ước lượng khoảng cách tương đối (Depth Mapping Logic)
*   Để ước lượng mức gần/xa của xe phía trước, hệ thống lấy tọa độ hộp bao của xe từ Faster R-CNN.
*   Cắt phân vùng tương ứng trên bản đồ độ sâu của MiDaS (tập trung vào 1/3 phía dưới của hộp bao vì đây là điểm tiếp xúc bánh xe gần đúng của phương tiện với mặt đường).
*   Tính giá trị trung vị (Median Depth) của vùng này để tránh nhiễu và tạo chỉ số khoảng cách tương đối:
    $$d_{rel} = \frac{1000.0}{\text{Depth}_{\text{median}} + 10^{-5}}$$
*   `d_rel` chỉ là chỉ số heuristic không có đơn vị. Muốn suy ra mét cần camera calibration và dữ liệu ground truth.

### 2.3. Logic quyết định cảnh báo va chạm (Collision Alert Decision)
*   Sử dụng hàm kiểm tra điểm trong đa giác (`cv2.pointPolygonTest`) để xác định xem điểm tiếp xúc bánh xe của phương tiện phía trước có nằm trong đa giác làn đường động đã dựng hay không.
*   Nếu nằm trong làn hiện tại của xe chủ:
    *   Kiểm tra chỉ số tương đối $d_{rel}$.
    *   Nếu chỉ số thỏa điều kiện cảnh báo heuristic của demo thì hiển thị banner đỏ và đổi màu khung bao.
    *   Ngược lại, hiển thị khung bao màu xanh lá/vàng bình thường.
