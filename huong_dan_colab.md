# HƯỚNG DẪN CHI TIẾT CHẠY PIPELINE TRÊN GOOGLE COLAB

Tài liệu này hướng dẫn từng bước (Step-by-step) để bạn đưa dự án lên Google Colab và tận dụng GPU T4 miễn phí nhằm xử lý video giao thông thời gian thực với tốc độ cực cao (>30 FPS), sau đó xuất ra file video thành phẩm để tải về máy tính báo cáo.

---

## BƯỚC 1: CHUẨN BỊ VÀ ĐƯA DỰ ÁN LÊN GOOGLE DRIVE

Cách đơn giản và ổn định nhất để chạy trên Google Colab là lưu trữ dự án trên Google Drive của bạn:

1. **Nén thư mục dự án**: 
   * Hãy nén thư mục `traffic-scene-understanding-btl` của bạn thành file `.zip` (ví dụ: `project.zip`).
   * *Lưu ý*: Hãy đảm bảo đã bao gồm thư mục `weights` (chứa `weights/UNET_resnet50_road/best_model.pth`), thư mục `data/sample_videos` (chứa các video kiểm thử như `video1.mp4`, `video2.mp4`), và thư mục `MidasDepthEstimation`.
2. **Tải lên Google Drive**:
   * Mở Google Drive của bạn.
   * Tải file `project.zip` lên một thư mục trên Google Drive (Ví dụ: tạo thư mục tên là `BTL_XLA` rồi tải file zip lên đó).

---

## BƯỚC 2: MỞ GOOGLE COLAB VÀ CẤU HÌNH GPU

1. Truy cập vào trang web: [Google Colab](https://colab.research.google.com/).
2. Chọn **New Notebook** (Sổ tay mới) để tạo một file làm việc mới.
3. Kích hoạt phần cứng GPU (Bắt buộc để chạy nhanh):
   * Ở thanh menu phía trên, chọn: **Runtime** -> **Change runtime type** (Thay đổi loại thời gian chạy).
   * Ở mục **Hardware accelerator** (Bộ tăng tốc phần cứng), chọn **T4 GPU** (Đây là GPU miễn phí của Google, rất mạnh cho xử lý ảnh).
   * Nhấn **Save** (Lưu).

---

## BƯỚC 3: CÁC LỆNH THỰC THI TRÊN GOOGLE COLAB (COPY-PASTE VÀO CÁC CELL)

Hãy tạo các ô mã nguồn (Code cell) trong Google Colab và chạy lần lượt các lệnh sau:

### Ô số 1: Kết nối Google Colab với Google Drive của bạn
Chạy cell này và làm theo hướng dẫn trên màn hình để cấp quyền truy cập Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Ô số 2: Giải nén thư mục dự án vào môi trường chạy của Colab
Giải nén dự án trực tiếp vào ổ đĩa ảo của Colab giúp tốc độ đọc/ghi file nhanh hơn gấp nhiều lần so với đọc trực tiếp từ Drive:
```bash
# Thay đổi đường dẫn 'My Drive/BTL_XLA/project.zip' cho đúng với vị trí bạn lưu trên Drive của mình
!unzip -q "/content/drive/My Drive/BTL_XLA/project.zip" -d "/content/"
```
*(Sau khi giải nén xong, bạn sẽ thấy thư mục `traffic-scene-understanding-btl` xuất hiện ở mục Files bên tay trái Colab)*.

### Ô số 3: Cài đặt các thư viện cần thiết
Di chuyển vào thư mục code và cài đặt các package phụ thuộc:
```bash
%cd /content/traffic-scene-understanding-btl
!pip install -r requirements.txt
```

### Ô số 4: Thực thi Pipeline xử lý video ẩn danh (Headless) và Lưu video đầu ra
Chạy tập lệnh chính để thực hiện nhận diện làn đường, phát hiện phương tiện, đo khoảng cách và ghi video thành phẩm:
```bash
# Lệnh chạy video 1 (ho-chi-minh-road-traffic.mp4):
!python -m utils.main --video_path data/sample_videos/ho-chi-minh-road-traffic.mp4 --headless --save_video output_video1.mp4

# Hoặc lệnh chạy video 3 (video3lightneed.mp4 - đường hai chiều, bật cờ --full_road):
# !python -m utils.main --video_path data/sample_videos/video3lightneed.mp4 --headless --full_road --save_video output_video3.mp4
```

> [!NOTE]
> *   Cờ `--headless` đảm bảo chương trình chạy nền không mở cửa sổ OpenCV (vì Colab không hỗ trợ hiển thị màn hình trực tiếp).
> *   Cờ `--save_video output_video1.mp4` sẽ ghi toàn bộ luồng video xử lý HUD thành file `output_video1.mp4`.
> *   Nhờ có **GPU T4**, tốc độ xử lý sẽ đạt mức **~30 FPS** (Thời gian thực), video 10 giây chỉ mất khoảng 10 giây để xử lý xong.

---

## BƯỚC 4: TẢI VIDEO THÀNH PHẨM VỀ MÁY TÍNH VÀ HOÀN THÀNH

1. Khi Ô số 4 chạy xong và in ra dòng: `[SUCCESS] Da ghi xong video output vao: output_video1.mp4`.
2. Bạn nhìn sang **Thanh công cụ bên trái** của Colab, chọn biểu tượng **Thư mục (Files)**.
3. Tìm file **`output_video1.mp4`** nằm ngay trong thư mục `traffic-scene-understanding-btl`.
4. Click vào biểu tượng **3 dấu chấm** bên cạnh file đó -> Chọn **Download** (Tải xuống).
5. Mở file video vừa tải về trên máy tính của bạn: Video sẽ chạy cực kỳ mượt mà, hiển thị đầy đủ màn hình ghép Dashboard 3 ô (HUD kết quả + Phân đoạn U-Net màu tím + Độ sâu MiDaS) để bạn chèn trực tiếp vào slide thuyết trình BTL.
