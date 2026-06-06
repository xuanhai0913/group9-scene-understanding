# Implement

## Huong trien khai tong quan

Qua trinh implement nen lam theo tung module nho de de chia viec va de debug.

Thu tu trien khai:

1. Chuan bi anh dau vao.
2. Tien xu ly anh.
3. Chay semantic segmentation.
4. Chay depth estimation.
5. Ket hop ket qua.
6. Tao hinh demo va bao cao.

## Cau truc thu muc de xuat

```text
group9-scene-understanding/
├── data/
│   ├── raw/
│   └── samples/
├── outputs/
│   ├── segmentation/
│   ├── depth/
│   └── fusion/
├── notebooks/
│   └── demo_scene_understanding.ipynb
├── src/
│   ├── preprocess.py
│   ├── segmentation.py
│   ├── depth.py
│   ├── fusion.py
│   └── visualize.py
├── README.md
└── requirements.txt
```

Giai doan hien tai chua can tao het source code. Cau truc nay la de nhom co huong khi bat dau code.

## Module preprocess

Nhiem vu:

- Doc anh bang OpenCV hoac PIL.
- Resize anh ve kich thuoc phu hop.
- Doi mau BGR/RGB neu can.
- Normalize anh truoc khi dua vao model.

Output:

- Anh da tien xu ly.
- Shape anh de kiem tra.

## Module segmentation

Nhiem vu:

- Load model segmentation.
- Dua anh vao model.
- Lay mask du doan.
- Gan mau cho tung class.
- Tao overlay mask len anh goc.

Output:

- Segmentation mask.
- Segmentation overlay.

Neu dung U-Net tu train:

- Can chuan bi dataset va label.
- Can chia train/validation.
- Can metric nhu pixel accuracy va mean IoU.

Neu dung pretrained:

- Can noi ro model da duoc huan luyen tren dataset nao.
- Tap trung vao inference va giai thich ket qua.

## Module depth

Nhiem vu:

- Load model MiDaS.
- Chuyen anh sang input tensor.
- Chay inference.
- Resize depth map ve kich thuoc anh goc.
- Normalize depth map de hien thi.

Output:

- Depth map.
- Depth colormap.

Luu y:

- Depth cua MiDaS la relative depth.
- Khi thuyet trinh nen noi la uoc luong gan xa tuong doi.

## Module fusion

Nhiem vu:

- Nhan segmentation mask va depth map.
- Tinh depth trung binh theo tung class neu can.
- Tao nhan xet ve ngu canh.
- Lam noi bat doi tuong gan camera.

Vi du output:

```text
road: chiem phan lon phia duoi anh, depth thay doi theo phoi canh.
car: co mot vung gan camera hon so voi nen.
sky: thuong nam phia tren anh va la vung xa.
```

## Notebook demo

Notebook nen co cac cell:

- Import thu vien.
- Doc anh mau.
- Hien thi anh goc.
- Chay segmentation.
- Hien thi mask va overlay.
- Chay depth estimation.
- Hien thi depth map.
- Ket hop va nhan xet.
- Luu output.

## Chia viec cho nhom 6 nguoi

Goi y chia viec:

- Hai: leader, tong quan de tai, AI workflow, pipeline va doi dap.
- Thanh vien 1: dataset Cityscapes/KITTI, mo ta input/output.
- Thanh vien 2: semantic segmentation, U-Net, mask va metric.
- Thanh vien 3: depth estimation, MiDaS, relative depth.
- Thanh vien 4: fusion, visualization, overlay va output demo.
- Thanh vien 5: bao cao, slide, test case va cau hoi phan bien.

## Cau tom tat khi thuyet trinh

Nhom em trien khai theo pipeline ro rang: doc anh, tien xu ly, chay segmentation de phan vung doi tuong, chay depth de uoc luong gan xa, sau do ket hop hai ket qua va tao hinh demo. Cach chia module giup nhom de debug va de giai thich tung phan khi bao ve.

