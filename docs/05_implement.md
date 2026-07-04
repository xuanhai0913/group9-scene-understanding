# README 5: Implementation (Triển khai mã nguồn)

Tài liệu này hướng dẫn chi tiết về cấu trúc mã nguồn và cách cài đặt dự án.

---

## 1. Cấu trúc thư mục dự án

```
traffic-scene-understanding-btl/
│
├── config/
│   ├── train_config.yaml         # Lưu cấu hình tham số hệ thống (nguỡng, đường dẫn)
│   └── train_config_cityscapes.yaml # Cấu hình tham số đánh giá và kích thước ảnh
│
├── data/
│   ├── kitti/                    # Chứa tập dữ liệu KITTI Road (training, testing)
│   ├── test_images/              # Chứa các ảnh kết quả phân vùng sau khi chạy eval
│   └── sample_videos/            # Chứa các video kiểm thử (hanoi, hochiminh, video3...)
│
├── utils/
│   ├── __init__.py               # Đăng ký các module
│   ├── main.py                   # Triển khai pipeline chính (chạy video, vẽ HUD, cảnh báo, luồng quang học)
│   ├── video_loader.py           # Module đọc video và camera đa luồng
│   ├── fusion.py                 # Module lọc đường chân trời ROI, trích xuất Contours và hợp nhất dữ liệu
│   ├── tracker.py                # Module theo dõi vật thể và gán ID tương đối
│   ├── visualization.py          # Module vẽ khung bao HUD, radar cảnh báo, bảng dashboard
│   ├── model.py                  # Định nghĩa mạng U-Net với backbone ResNet50
│   ├── dataset.py                # Module quản lý tải dữ liệu huấn luyện
│   ├── trainer.py                # Định nghĩa lớp Meter đo đạc chỉ số IoU/Dice khi huấn luyện
│   └── kitti_lane_utils.py       # Module xử lý dữ liệu ảnh KITTI
│
├── weights/
│   └── UNET_resnet50_road/
│       └── best_model.pth        # File trọng số mô hình U-Net ResNet50 tối ưu mới nhất
│
├── eval.py                       # Script chạy đánh giá mô hình U-Net trên tập Validation
├── generate_heatmap.py           # Script tự động vẽ biểu đồ Confusion Matrix Heatmap
└── requirements.txt              # Danh sách các thư viện cần cài đặt
```

---

## 2. Hướng dẫn cài đặt và thiết lập môi trường

### Bước 1: Cài đặt các thư viện phụ thuộc
Yêu cầu Python từ phiên bản 3.8 trở lên. Cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```

### Bước 2: Chuẩn bị model weights và dữ liệu

Các file lớn không được lưu trong Git. Tải tài nguyên từ [Google Drive của nhóm](https://drive.google.com/drive/folders/1Q4kjK8xg9h7dO16AzeHo5A71DnF5Idr1?hl=vi), sau đó đặt checkpoint U-Net tại:

```text
weights/UNET_resnet50_road/best_model.pth
```

MiDaS TFLite sẽ được tải tự động vào `models/midasModel.tflite` trong lần chạy đầu. Dữ liệu KITTI, Cityscapes và media demo cần được đặt đúng các đường dẫn trong `config/train_config.yaml`.

### Bước 3: Kiểm tra cấu hình hệ thống
Mở file `config/train_config.yaml` để kiểm tra các tham số quan trọng:
*   `EVAL.base_threshold`: Đặt ở mức `-2.5` để tối ưu Recall và IoU cho U-Net.
*   `EVAL.device`: Có thể tùy chọn `'cpu'` hoặc `'cuda'` nếu máy có hỗ trợ GPU Nvidia.
