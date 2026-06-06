# Tech Solutions

## Hướng kỹ thuật tổng quan

Dự án dùng hai nhánh xử lý song song:

- Nhánh semantic segmentation để phân vùng ngữ nghĩa.
- Nhánh depth estimation để ước lượng độ sâu.

Cả hai nhánh đều nhận cùng một ảnh đường phố làm input. Sau đó hệ thống kết hợp hai output để tạo kết quả phân tích ngữ cảnh giao thông.

## Dataset

### Cityscapes

Cityscapes phù hợp với semantic segmentation trong môi trường đô thị. Dataset này có ảnh đường phố và nhãn theo pixel cho các lớp như road, sidewalk, car, person, building, vegetation và sky.

Vai trò trong dự án:

- Là nguồn dữ liệu tham khảo cho segmentation.
- Có thể dùng ảnh mẫu và ground truth để giải thích bài toán.
- Nếu train/fine-tune, có thể dùng subset nhỏ để giảm tải.

### KITTI

KITTI phù hợp với ảnh giao thông và các bài toán liên quan đến xe tự hành, trong đó có depth. Dataset này có ảnh đường phố từ camera gắn trên xe và dữ liệu depth/liên quan đến không gian 3D.

Vai trò trong dự án:

- Là nguồn ảnh giao thông tham khảo.
- Dùng để giải thích bài toán depth estimation.
- Có thể dùng ảnh mẫu để chạy MiDaS và so sánh trực quan.

## Model Semantic Segmentation

### U-Net

U-Net có cấu trúc encoder-decoder. Encoder trích xuất đặc trưng của ảnh, decoder khôi phục lại kích thước để tạo mask theo pixel. Skip connection giúp giữ lại thông tin chi tiết, đặc biệt là vùng biên của đối tượng.

Lý do chọn U-Net:

- Dễ giải thích.
- Phù hợp với segmentation theo pixel.
- Có kiến trúc rõ ràng.
- Có thể train trên subset nhỏ nếu cần.

Hạn chế:

- Nếu train từ đầu có thể cần dữ liệu và thời gian.
- Kết quả có thể kém model hiện đại nếu dataset ít.

Phương án thay thế:

- DeepLabV3.
- SegFormer.
- Model segmentation pretrained trong PyTorch/Torchvision.

## Model Depth Estimation

### MiDaS

MiDaS là model ước lượng depth từ một ảnh đơn. Output là depth map cho biết vùng nào gần hơn hoặc xa hơn trong ảnh.

Lý do chọn MiDaS:

- Có model pretrained.
- Không cần camera stereo.
- Không cần LiDAR.
- Dễ demo trên ảnh đường phố.
- Output trực quan.

Hạn chế:

- Thường cho relative depth, không phải khoảng cách tuyệt đối.
- Có thể sai ở vùng phản chiếu, vật thể mỏng hoặc cảnh quá khác dữ liệu train.

## Thư viện đề xuất

- Python: ngôn ngữ chính.
- OpenCV: đọc ảnh, resize, xử lý ảnh cơ bản, tạo overlay.
- NumPy: xử lý ma trận ảnh.
- Matplotlib: hiển thị ảnh, mask và depth map.
- PyTorch: chạy model deep learning.
- Torchvision: transform ảnh, model pretrained nếu cần.
- PIL/Pillow: hỗ trợ đọc ảnh và thao tác ảnh đơn giản.

## Kiến trúc hệ thống

```text
Input image
    |
    v
Preprocessing
    | resize, normalize, convert color
    |
    +--------------------------+
    |                          |
    v                          v
Segmentation Model          Depth Model
    |                          |
    v                          v
Semantic Mask               Depth Map
    |                          |
    +------------+-------------+
                 |
                 v
              Fusion
                 |
                 v
       Visualization + Report
```

## Tech Solution theo từng module

### Preprocessing

- Đọc ảnh.
- Chuyển RGB/BGR nếu cần.
- Resize ảnh.
- Normalize ảnh.
- Chuyển ảnh sang tensor nếu dùng PyTorch.

### Segmentation

- Load U-Net hoặc model pretrained.
- Inference ảnh đầu vào.
- Lấy class có xác suất cao nhất cho từng pixel.
- Tạo mask màu.
- Overlay lên ảnh gốc.

### Depth

- Load MiDaS.
- Transform ảnh đầu vào.
- Inference depth.
- Resize depth map về kích thước ảnh gốc.
- Normalize để hiển thị.

### Fusion

- Căn chỉnh segmentation mask và depth map cùng kích thước.
- Kết hợp hai output theo pixel.
- Tính depth trung bình theo lớp nếu cần.
- Tạo nhận xét về scene.

## Câu tóm tắt khi thuyết trình

Nhóm em chọn Cityscapes cho segmentation vì dataset này có nhãn pixel trong cảnh đường phố, và chọn KITTI cho depth vì dữ liệu gắn với bài toán giao thông. Về model, U-Net giúp giải thích segmentation theo pixel, còn MiDaS giúp ước lượng độ sâu từ một ảnh đơn. Hai kết quả được kết hợp để phân tích ngữ cảnh giao thông.

