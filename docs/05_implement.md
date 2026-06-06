# Implement

## Hướng triển khai tổng quan

Quá trình implement nên chia thành các module nhỏ để dễ phân công và dễ debug.

Thứ tự triển khai:

- Chuẩn bị ảnh đầu vào.
- Tiền xử lý ảnh.
- Chạy semantic segmentation.
- Chạy depth estimation.
- Kết hợp kết quả.
- Tạo hình demo và báo cáo.

## Cấu trúc thư mục đề xuất

```text
group9-scene-understanding/
├── data/
│   ├── raw/
│   └── samples/
├── docs/
├── notebooks/
│   └── demo_scene_understanding.ipynb
├── outputs/
│   ├── segmentation/
│   ├── depth/
│   └── fusion/
├── src/
│   ├── preprocess.py
│   ├── segmentation.py
│   ├── depth.py
│   ├── fusion.py
│   └── visualize.py
├── requirements.txt
└── README.md
```

Giai đoạn tài liệu hiện tại chưa cần tạo hết source code. Cấu trúc trên dùng để nhóm có hướng khi bắt đầu code.

## Module Preprocessing

Nhiệm vụ:

- Đọc ảnh bằng OpenCV hoặc PIL.
- Chuyển ảnh về RGB nếu cần.
- Resize ảnh về kích thước model yêu cầu.
- Normalize ảnh.
- Chuyển ảnh sang tensor nếu dùng PyTorch.

Output:

- Ảnh đã tiền xử lý.
- Shape ảnh.
- Tensor đầu vào cho model.

## Module Segmentation

Nhiệm vụ:

- Load model segmentation.
- Đưa ảnh vào model.
- Lấy mask dự đoán.
- Gán màu cho từng class.
- Tạo overlay mask lên ảnh gốc.

Output:

- Semantic mask.
- Colored mask.
- Segmentation overlay.

Nếu dùng U-Net tự train:

- Chuẩn bị ảnh và nhãn.
- Chia train/validation.
- Train trên subset nhỏ.
- Đánh giá bằng pixel accuracy hoặc mean IoU.

Nếu dùng pretrained:

- Ghi rõ model đã được huấn luyện trên dataset nào.
- Tập trung vào inference và giải thích output.

## Module Depth

Nhiệm vụ:

- Load MiDaS.
- Transform ảnh đầu vào.
- Chạy inference.
- Resize depth map về kích thước ảnh gốc.
- Normalize depth map để hiển thị.

Output:

- Depth map.
- Depth colormap.

Lưu ý:

- Depth của MiDaS là relative depth.
- Khi thuyết trình nên nói là ước lượng gần/xa tương đối.

## Module Fusion

Nhiệm vụ:

- Nhận segmentation mask và depth map.
- Căn chỉnh kích thước hai output.
- Tính depth trung bình theo từng class nếu cần.
- Tạo nhận xét về ngữ cảnh.
- Làm nổi bật đối tượng gần camera.

Ví dụ output:

```text
road: chiếm phần lớn phía dưới ảnh, depth thay đổi theo phối cảnh.
car: có một vùng gần camera hơn so với nền.
sky: thường nằm phía trên ảnh và là vùng xa.
```

## Notebook demo

Notebook nên có các cell:

- Import thư viện.
- Đọc ảnh mẫu.
- Hiển thị ảnh gốc.
- Chạy segmentation.
- Hiển thị mask và overlay.
- Chạy depth estimation.
- Hiển thị depth map.
- Kết hợp segmentation và depth.
- Lưu output.
- In checklist kết quả.

## Chia việc cho nhóm 6 người

Gợi ý chia việc:

- Hải: leader, tổng quan đề tài, AI workflow, pipeline và đối đáp.
- Thành viên 1: dataset Cityscapes/KITTI, mô tả input/output.
- Thành viên 2: semantic segmentation, U-Net, mask và metric.
- Thành viên 3: depth estimation, MiDaS, relative depth.
- Thành viên 4: fusion, visualization, overlay và output demo.
- Thành viên 5: báo cáo, slide, test case và câu hỏi phản biện.

## Câu tóm tắt khi thuyết trình

Nhóm em triển khai theo pipeline rõ ràng: đọc ảnh, tiền xử lý, chạy segmentation để phân vùng đối tượng, chạy depth để ước lượng gần xa, sau đó kết hợp hai kết quả và tạo hình demo. Cách chia module giúp nhóm dễ debug và dễ giải thích từng phần khi bảo vệ.

