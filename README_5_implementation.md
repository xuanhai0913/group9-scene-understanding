# README 5: Implementation (Triển khai mã nguồn)

Tài liệu này hướng dẫn chi tiết về cấu trúc mã nguồn và cách cài đặt dự án.

---

## 1. Cấu trúc thư mục dự án

```
traffic-scene-understanding-btl/
│
├── config/
│   └── train_config.yaml         # Lưu cấu hình tham số hệ thống (ngưỡng, đường dẫn)
│
├── data/
│   ├── kitti/                    # Chứa tập dữ liệu KITTI Road (training, testing)
│   ├── test_images/              # Chứa các ảnh kết quả phân vùng sau khi chạy eval
│   └── sample_videos/            # Chứa các video kiểm thử (video1, video2, video3...)
│
├── utils/
│   ├── __init__.py               # Đăng ký các module
│   ├── main.py                   # Triển khai pipeline chính (chạy video, vẽ HUD, cảnh báo)
│   ├── video_loader.py           # Module đọc video và camera
│   ├── kitti_lane_utils.py       # Module xử lý dữ liệu ảnh KITTI
│   └── trainer.py                # Định nghĩa lớp Meter đo đạc chỉ số IoU/Dice
│
├── weights/
│   └── unet_best.pth             # File trọng số mô hình U-Net đã được train sẵn
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

### Bước 2: Kiểm tra cấu hình hệ thống
Mở file cấu hình [config/train_config.yaml](file:///d:/BTL_XLA/traffic-scene-understanding-btl/config/train_config.yaml) để kiểm tra các tham số quan trọng:
*   `EVAL.base_threshold`: Đặt ở mức `-2.5` để tối ưu Recall và IoU cho U-Net.
*   `EVAL.device`: Có thể tùy chọn `'cpu'` hoặc `'cuda'` nếu máy có hỗ trợ GPU Nvidia.

