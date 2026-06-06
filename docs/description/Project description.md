# Project Description

## Tên dự án

**Traffic Scene Understanding using Semantic Segmentation and Depth Estimation**

Tên tiếng Việt:

**Phân tích ngữ cảnh giao thông dựa trên Semantic Segmentation và Depth Estimation**

## Vấn đề cần giải quyết

Trong ảnh đường phố, con người có thể dễ dàng nhận ra đâu là mặt đường, đâu là xe, đâu là bầu trời và vật thể nào đang ở gần hoặc xa. Tuy nhiên, với máy tính, ảnh chỉ là một ma trận pixel. Dự án này xây dựng một pipeline giúp máy tính hiểu cảnh giao thông ở hai mức:

- Hiểu **vùng nào là gì** thông qua semantic segmentation.
- Hiểu **vùng nào gần/xa** thông qua depth estimation.

Khi kết hợp hai nguồn thông tin này, hệ thống có thể phân tích ngữ cảnh giao thông tốt hơn so với chỉ nhìn ảnh màu gốc.

## Input

Input là ảnh đường phố, có thể lấy từ:

- Cityscapes.
- KITTI.
- Ảnh đường phố dùng để demo.

Yêu cầu input:

- Có mặt đường, xe, bầu trời, vỉa hè hoặc các đối tượng giao thông.
- Ảnh đủ rõ để segmentation và depth estimation có kết quả quan sát được.

## Output

Output chính:

- Semantic segmentation mask.
- Segmentation overlay.
- Depth map.
- Fusion result.
- Nhận xét ngắn về ngữ cảnh giao thông.

Ví dụ output mong muốn:

```text
Ảnh gốc: cảnh đường phố.
Segmentation: road, car, sky, sidewalk.
Depth map: vùng gần/xa tương đối.
Fusion: xe bên trái gần camera hơn nền phía xa; road chiếm vùng phía trước.
```

## Scope

Phạm vi dự án:

- Xử lý ảnh đường phố tĩnh.
- Dùng model pretrained hoặc baseline model phù hợp.
- Tạo demo trực quan.
- Giải thích pipeline, model, dataset và hạn chế.
- Ghi lại quy trình dùng AI đúng cách.
- Tạo checklist test và edge cases.

## Out Of Scope

Không làm trong giai đoạn đầu:

- Không xây dựng xe tự hành hoàn chỉnh.
- Không xử lý real-time video nếu chưa đủ thời gian.
- Không tracking người/xe theo nhiều frame.
- Không phát hiện vi phạm giao thông.
- Không đo khoảng cách mét tuyệt đối nếu không có calibration.
- Không train toàn bộ dataset lớn nếu máy không đủ.

## Features List

Tính năng cần có:

- Đọc ảnh đường phố.
- Tiền xử lý ảnh.
- Chạy semantic segmentation.
- Tạo segmentation mask.
- Overlay segmentation lên ảnh gốc.
- Chạy depth estimation.
- Tạo depth map.
- Normalize depth map để hiển thị.
- Kết hợp segmentation và depth.
- Lưu output.
- Tạo notebook/script demo.
- Ghi log quá trình làm việc với AI.
- Tạo test checklist.

## Edge Cases

Cần lưu ý:

- Ảnh thiếu sáng hoặc ban đêm.
- Ảnh có mưa, sương mù hoặc lóa sáng.
- Xe bị che khuất.
- Vạch đường mờ.
- Đường và vỉa hè có màu gần giống nhau.
- Ảnh có nhiều vật thể nhỏ.
- Depth map sai ở vùng phản chiếu, kính hoặc bề mặt bóng.
- Model nhầm lớp nếu ảnh khác nhiều so với dataset huấn luyện.

## Tech Solution

Ngôn ngữ và thư viện:

- Python.
- OpenCV.
- NumPy.
- Matplotlib.
- PyTorch.

Dataset:

- Cityscapes cho semantic segmentation.
- KITTI cho depth/ảnh giao thông.

Model:

- U-Net hoặc pretrained segmentation model cho semantic segmentation.
- MiDaS cho monocular depth estimation.

Pipeline:

```text
Input image
    -> Preprocessing
    -> Semantic Segmentation
    -> Depth Estimation
    -> Fusion
    -> Visualization + Report
```

## AI Usage

AI được dùng để hỗ trợ:

- Làm rõ requirement.
- Góp ý scope và out of scope.
- Gợi ý features.
- So sánh dataset/model.
- Rà logic segmentation + depth.
- Lập test checklist.
- Ghi log thảo luận.

AI không được dùng để:

- Làm toàn bộ dự án thay nhóm.
- Che giấu việc nhóm chưa hiểu code.
- Tạo output mà nhóm không kiểm tra.

