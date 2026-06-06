# Tech Solutions

## Huong ky thuat tong quan

De tai su dung hai nhanh xu ly chinh:

- Semantic segmentation bang U-Net hoac model segmentation pretrained.
- Depth estimation bang MiDaS hoac model depth pretrained.

Hai nhanh nay nhan cung mot anh dau vao, sau do sinh ra hai output rieng: segmentation mask va depth map.

## Dataset

### Cityscapes

Cityscapes phu hop voi semantic segmentation trong moi truong do thi. Dataset nay co anh duong pho va nhan ngu nghia theo pixel. Cac lop nhu road, sidewalk, car, person, building, sky rat phu hop voi de tai.

Vai tro trong de tai:

- Dung de minh hoa segmentation.
- Neu train/fine-tune, co the dung subset nho.
- Neu khong train, dung lam nguon anh mau va ground truth tham khao.

### KITTI

KITTI phu hop voi anh giao thong va depth estimation. Dataset nay co anh chup tu xe va du lieu depth lien quan den moi truong lai xe.

Vai tro trong de tai:

- Dung lam nguon anh duong pho.
- Dung tham khao cho bai toan depth.
- Co the dung anh mau de chay MiDaS va so sanh truc quan.

## Model segmentation

### U-Net

U-Net co cau truc encoder-decoder. Encoder trich xuat dac trung anh, decoder khoi phuc lai kich thuoc de tao mask theo pixel. Skip connection giup giu lai thong tin chi tiet, dac biet la bien doi tuong.

Ly do chon U-Net:

- De giai thich voi giao vien.
- Phu hop bai toan segmentation.
- Co nhieu source code va tai lieu tham khao.
- Co the train tren subset nho neu can.

Neu can ket qua tot hon va nhanh hon, nhom co the dung model pretrained nhu DeepLabV3 hoac SegFormer, nhung trong bao cao van co the lay U-Net lam kien truc nen tang de giai thich.

## Model depth

### MiDaS

MiDaS la model uoc luong depth tu mot anh don. Model nay tao ra depth map bieu dien do sau tuong doi cua canh.

Ly do chon MiDaS:

- Co model pretrained.
- Chay duoc voi anh don, khong can stereo camera.
- Output truc quan, de demo.
- Phu hop voi bai toan scene understanding.

Han che:

- Depth thuong la tuong doi, khong phai khoang cach met tuyet doi.
- Ket qua phu thuoc vao anh dau vao va model pretrained.

## Thu vien de xuat

Thu vien co the dung:

- Python: ngon ngu chinh.
- OpenCV: doc anh, resize, tien xu ly, hien thi overlay.
- NumPy: xu ly mang anh.
- Matplotlib: hien thi anh, mask va depth map.
- PyTorch: chay U-Net/MiDaS.
- Torchvision: ho tro transform anh.

## Kien truc he thong

```text
Input image
    |
    +-- Preprocessing
    |       resize, normalize
    |
    +-- Segmentation model
    |       output: semantic mask
    |
    +-- Depth model
    |       output: depth map
    |
    +-- Fusion
            overlay, analysis, visualization
```

## Cau tom tat khi thuyet trinh

Nhom em chon Cityscapes cho segmentation vi no co nhan pixel trong moi truong do thi, va KITTI cho depth vi no gan voi bai toan lai xe. Ve model, U-Net de giai thich segmentation theo pixel, MiDaS de uoc luong do sau tu mot anh don. Hai ket qua nay duoc ket hop de phan tich ngu canh giao thong.

