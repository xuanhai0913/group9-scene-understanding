# Prompt Workflow Draft

File này là **mẫu quy trình prompt tham khảo**, không phải log thực tế. Khi nhóm làm việc thật với Agent AI/Copilot, nhóm có thể dựa vào đây để đặt câu hỏi theo từng phần, sau đó mới ghi nội dung thật vào `logdiscusssion.md` nếu nhóm trưởng cho phép.

## Nguyên tắc dùng prompt

Không hỏi AI theo kiểu:

```text
Làm hết bài tập lớn này cho tôi.
```

Nên hỏi theo kiểu:

```text
Nhóm em đã tự xác định yêu cầu như sau..., bạn kiểm tra giúp cách hiểu này đã đủ chưa.
```

Như vậy AI đóng vai trò kiểm tra, góp ý và phản biện, còn nhóm vẫn là người hiểu bài và ra quyết định.

## Knowledge base đưa cho AI trước

```text
Nhóm 9 làm đề tài Traffic Scene Understanding.
Input: ảnh đường phố.
Output: semantic segmentation mask, segmentation overlay, depth map và fusion result.
Dataset tham khảo: Cityscapes cho segmentation, KITTI cho depth.
Model dự kiến: U-Net hoặc pretrained segmentation model, MiDaS cho depth estimation.
Scope: demo phân tích ngữ cảnh giao thông từ ảnh tĩnh.
Out of scope: không xây dựng xe tự hành hoàn chỉnh, không đo khoảng cách mét tuyệt đối nếu không có calibration.
```

## User's Requirement

Prompt mẫu:

```text
Nhóm em đang chọn đề tài phân tích ngữ cảnh giao thông từ ảnh đường phố. Em tự hiểu yêu cầu là: input là một ảnh đường phố, output gồm semantic segmentation mask để biết từng pixel thuộc lớp nào và depth map để biết vùng nào gần/xa tương đối. Bạn kiểm tra giúp cách hiểu này đã đúng chưa, còn thiếu yêu cầu quan trọng nào không, và nên giới hạn phạm vi thế nào cho phù hợp bài tập lớn.
```

Mục đích:

- Kiểm tra nhóm có hiểu đúng bài toán không.
- Chốt input/output.
- Chốt giới hạn phạm vi.

## Features

Prompt mẫu:

```text
Sau khi chốt yêu cầu, nhóm em dự kiến các feature chính gồm: đọc ảnh đường phố, tiền xử lý ảnh, chạy semantic segmentation, tạo mask màu, overlay mask lên ảnh gốc, chạy depth estimation, tạo depth map, kết hợp segmentation với depth và lưu output. Bạn góp ý giúp feature list này đã đủ chưa, feature nào nên ưu tiên làm trước để có demo rõ ràng.
```

Mục đích:

- Tách bài toán thành các tính năng nhỏ.
- Ưu tiên feature quan trọng.
- Tránh ôm quá rộng.

## Tech Solutions

Prompt mẫu:

```text
Về tech solution, nhóm em dự kiến dùng Python, OpenCV, NumPy, Matplotlib và PyTorch. Dataset tham khảo là Cityscapes cho semantic segmentation và KITTI cho depth. Model dự kiến là U-Net hoặc pretrained segmentation model cho segmentation, MiDaS cho depth estimation. Bạn phân tích giúp lựa chọn này có hợp lý không, ưu điểm và hạn chế của từng phần là gì.
```

Mục đích:

- Giải thích vì sao chọn công nghệ.
- Chuẩn bị câu trả lời khi giáo viên hỏi.
- Nhận ra hạn chế như MiDaS chỉ cho relative depth.

## Logic + AI

Prompt mẫu:

```text
Em muốn giải thích logic của đề tài như sau: semantic segmentation trả lời câu hỏi pixel thuộc lớp gì, depth estimation trả lời vùng đó gần hay xa, còn fusion là kết hợp class và depth để phân tích ngữ cảnh giao thông. Bạn kiểm tra giúp cách diễn đạt này có đúng không, và gợi ý cách nói rõ vai trò AI là hỗ trợ kiểm tra logic, không làm thay dự án.
```

Mục đích:

- Rà lại logic chính.
- Có câu giải thích ngắn để thuyết trình.
- Chứng minh AI được dùng có kiểm soát.

## Implement

Prompt mẫu:

```text
Nhóm em chuẩn bị implement demo bằng notebook. Dự kiến chia notebook thành các phần: import thư viện, đọc ảnh, preprocessing, segmentation, depth estimation, fusion, visualization và lưu output. Bạn góp ý giúp mỗi phần nên có input gì, output gì, cần in shape hoặc lưu file nào để dễ kiểm tra khi trình bày.
```

Mục đích:

- Chia notebook rõ ràng.
- Biết mỗi cell cần làm gì.
- Chuẩn bị output trực quan.

## Test

Prompt mẫu:

```text
Sau khi chạy demo, nhóm em muốn kiểm tra pipeline trước khi nộp. Bạn giúp lập checklist test cho preprocessing, segmentation, depth estimation và fusion. Nếu test fail thì nên kiểm tra gì trước? Lưu ý dự án dùng relative depth nên không đánh giá khoảng cách mét tuyệt đối.
```

Mục đích:

- Có checklist test.
- Biết debug lỗi thường gặp.
- Không nói quá khả năng của model.

## Câu nói với giáo viên

```text
Nhóm em không đưa nguyên đề tài cho AI làm từ đầu đến cuối. Trước khi hỏi AI, nhóm em chuẩn bị knowledge base gồm topic, input/output, dataset, model, scope và out of scope. Sau đó nhóm hỏi AI theo từng phần nhỏ: requirement, features, tech solution, logic, implement và test. AI chỉ hỗ trợ kiểm tra cách hiểu, góp ý hướng làm và dự đoán lỗi, còn quyết định cuối cùng và demo là do nhóm tự thực hiện.
```

