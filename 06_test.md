# Test

## Muc tieu test

Test dung de chung minh pipeline chay dung va output co y nghia. Vi de tai co hai nhanh segmentation va depth, test can kiem tra rieng tung nhanh va kiem tra ket qua ket hop.

## Test input

Can chuan bi nhieu anh duong pho:

- Anh co duong ro rang.
- Anh co xe gan va xe xa.
- Anh co bau troi va via he.
- Anh co nhieu doi tuong de segmentation kho hon.
- Anh thieu sang hoac co nhieu vat can neu muon test them.

## Test preprocessing

Can kiem tra:

- Anh doc duoc, khong bi None.
- Shape anh dung.
- Anh khong bi sai mau BGR/RGB.
- Resize khong lam meo ty le neu yeu cau giu ty le.
- Normalize dung format input cua model.

Neu fail:

- Kiem tra duong dan file.
- Kiem tra dinh dang anh.
- Kiem tra thu tu kenh mau.
- Kiem tra kich thuoc dau vao cua model.

## Test segmentation

Can kiem tra:

- Mask co cung kich thuoc voi anh hoac duoc resize ve dung kich thuoc.
- Moi pixel trong mask co class hop le.
- Cac lop chinh nhu road, sky, car co xuat hien hop ly.
- Overlay khong che hoan toan anh goc.

Metric neu co ground truth:

- Pixel accuracy.
- Mean IoU.
- Per-class IoU cho road, car, sky.

Neu khong co ground truth:

- Danh gia truc quan tren anh demo.
- So sanh voi anh goc va nhan xet loi ro rang.

## Test depth

Can kiem tra:

- Depth map tao ra duoc, khong bi rong.
- Depth map duoc resize ve dung kich thuoc anh goc.
- Vung gan va xa co su khac biet ro.
- Colormap hien thi de quan sat.

Dieu can noi ro:

- Depth cua MiDaS la gan xa tuong doi.
- Khong nen khang dinh khoang cach met neu khong co calibration hoac ground truth.

Metric neu co ground truth depth:

- MAE.
- RMSE.
- Abs Rel.

Neu khong co ground truth:

- Danh gia tuong quan truc quan: duong phia xa nho dan, bau troi/nen xa, xe gan noi bat hon.

## Test fusion

Can kiem tra:

- Segmentation mask va depth map duoc can chinh cung kich thuoc.
- Khi lay depth theo class, khong bi sai index pixel.
- Nhan xet ngu canh phu hop voi anh.

Vi du test:

```text
Neu vung car nam gan camera, depth cua vung car phai the hien muc gan hon so voi nen xa.
Neu vung sky nam phia tren anh, depth thuong the hien la vung xa.
Neu road nam phia duoi anh, depth thay doi theo huong xa dan ve phia chan troi.
```

## Checklist truoc khi nop

- Co it nhat 3 anh demo.
- Moi anh co anh goc, segmentation, depth va fusion.
- Co giai thich model dung gi va output la gi.
- Co noi ro han che relative depth.
- Co file notebook hoac script chay duoc.
- Co thu muc outputs luu ket qua.
- Co slide hoac bao cao tom tat pipeline.
- Co phan AI workflow va knowledge base.

## Cau tom tat khi thuyet trinh

Nhom em test theo tung phan: dau tien kiem tra anh dau vao va tien xu ly, sau do kiem tra segmentation mask, depth map, roi moi kiem tra phan ket hop. Neu co ground truth thi dung metric nhu IoU cho segmentation va RMSE cho depth. Neu khong co ground truth, nhom em danh gia truc quan tren nhieu anh va neu ro han che cua mo hinh.

