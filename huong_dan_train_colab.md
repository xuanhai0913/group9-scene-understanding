# HƯỚNG DẪN CHI TIẾT HUẤN LUYỆN (TRAIN) MÔ HÌNH U-NET ĐA LỚP TRÊN GOOGLE COLAB

Tài liệu này hướng dẫn bạn từng bước cách đưa dự án, chuẩn bị tập dữ liệu Cityscapes và huấn luyện mô hình U-Net nhận diện **4 lớp (Road, Sky, Car, Background)** trên Google Colab sử dụng GPU.

---

## BƯỚC 1: CHUẨN BỊ DỮ LIỆU CITYSCAPES TRÊN GOOGLE DRIVE
Vì tập dữ liệu Cityscapes rất nặng, bạn nên lưu trữ nó trên Google Drive cá nhân để Colab có thể đọc trực tiếp.

1. Đăng ký tài khoản trên trang chủ [Cityscapes Dataset](https://www.cityscapes-dataset.com/) (sử dụng email sinh viên để được phê duyệt nhanh).
2. Tải về 2 gói dữ liệu chính sau:
   * **`leftImg8bit_trainvaltest.zip`** (Dung lượng ~12GB - Ảnh thực tế).
   * **`gtFine_trainvaltest.zip`** (Dung lượng ~240MB - Nhãn phân đoạn chất lượng cao).
3. Tạo một thư mục tên là `BTL_XLA` trên Google Drive của bạn.
4. Tải trực tiếp 2 file zip này lên thư mục `BTL_XLA` trên Google Drive.

---

## BƯỚC 2: NÉN VÀ TẢI MÃ NGUỒN LÊN GOOGLE DRIVE
1. Hãy nén thư mục dự án `traffic-scene-understanding-btl` của bạn (sau khi đã cập nhật file cấu hình Cityscapes mới) thành file `project.zip`.
2. Tải file `project.zip` này lên thư mục `BTL_XLA` trên Google Drive.

---

## BƯỚC 3: MỞ GOOGLE COLAB VÀ CẤU HÌNH GPU T4
1. Truy cập [Google Colab](https://colab.research.google.com/) và tạo Sổ tay mới (New Notebook).
2. Kích hoạt GPU: Chọn **Runtime** -> **Change runtime type** -> Chọn **T4 GPU** -> Nhấn **Save**.

---

## BƯỚC 4: THỰC THI CÁC LỆNH TRÊN CELL CỦA COLAB

Hãy copy và chạy lần lượt các ô lệnh sau:

### Cell 1: Kết nối Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Giải nén mã nguồn dự án
```bash
!unzip -q "/content/drive/My Drive/BTL_XLA/project.zip" -d "/content/"
```

### Cell 3: Giải nén tập dữ liệu Cityscapes vào đúng thư mục dự án
Chúng ta sẽ giải nén trực tiếp vào thư mục `data/cityscapes` của dự án trên Colab để huấn luyện đạt tốc độ nhanh nhất:
```bash
# Tạo thư mục đích
!mkdir -p /content/traffic-scene-understanding-btl/data/cityscapes

# Giải nén ảnh gốc (leftImg8bit)
!unzip -q "/content/drive/My Drive/BTL_XLA/leftImg8bit_trainvaltest.zip" -d /content/traffic-scene-understanding-btl/data/cityscapes/

# Giải nén nhãn (gtFine)
!unzip -q "/content/drive/My Drive/BTL_XLA/gtFine_trainvaltest.zip" -d /content/traffic-scene-understanding-btl/data/cityscapes/
```

### Cell 4: Di chuyển vào thư mục và cài đặt thư viện
```bash
%cd /content/traffic-scene-understanding-btl
!pip install -r requirements.txt
```

### Cell 5: Bắt đầu huấn luyện mô hình U-Net đa lớp
Chạy tập lệnh train với cấu hình Cityscapes 4 lớp đã được thiết lập sẵn:
```bash
!python train.py --config_path config/train_config_cityscapes.yaml
```

---

## BƯỚC 5: TẢI TRỌNG SỐ ĐÃ HUẤN LUYỆN VỀ MÁY LOCAL
1. Sau khi quá trình train hoàn thành (khoảng 2 - 3.5 tiếng), file trọng số tốt nhất sẽ được lưu tại đường dẫn:
   `traffic-scene-understanding-btl/weights/UNET_resnet50_cityscapes/best_model.pth`.
2. Ở thanh công cụ bên trái của Colab, vào mục **Files** -> Tìm đến thư mục `weights/UNET_resnet50_cityscapes/`.
3. Nhấp vào dấu **3 chấm** bên cạnh file **`best_model.pth`** và chọn **Download** để tải về máy tính cá nhân.
4. Copy file này bỏ vào đúng thư mục `weights/UNET_resnet50_cityscapes/` trên máy local của bạn để chạy demo thực tế đa lớp!
