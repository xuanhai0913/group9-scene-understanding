# Logic + AI

## Logic xu ly bai toan

Logic chinh cua de tai la tach bai toan scene understanding thanh hai cau hoi:

1. Trong anh co nhung vung nao va moi vung thuoc lop gi?
2. Cac vung do gan hay xa camera?

Cau hoi thu nhat duoc giai bang semantic segmentation. Model nhan anh dau vao va du doan lop cho tung pixel. Ket qua la mask co cung kich thuoc voi anh dau vao, trong do moi pixel mang nhan lop nhu road, car, sky hoac sidewalk.

Cau hoi thu hai duoc giai bang depth estimation. Model nhan anh dau vao va tao depth map. Depth map giup nhan xet cau truc khong gian cua canh, vi vung gan camera va vung xa camera se co gia tri depth khac nhau.

Sau do, nhom ket hop hai ket qua. Vi du, neu segmentation cho thay mot vung la car va depth map cho thay vung do gan camera, he thong co the nhan xet do la doi tuong giao thong can chu y. Neu mot vung la sky va depth cho gia tri xa, ket qua hop ly voi ngu canh anh duong pho.

## Logic segmentation

Anh dau vao duoc resize va normalize. Model segmentation sinh ra ma tran du doan co kich thuoc gan voi anh. Moi pixel duoc gan vao lop co xac suat cao nhat.

Sau khi co mask, nhom gan mau cho tung lop de hien thi:

- road: mau xam hoac xanh la.
- car: mau do hoac cam.
- sky: mau xanh duong.
- sidewalk: mau vang hoac tim.

## Logic depth

Anh dau vao duoc dua vao model MiDaS. Model tra ve depth map. Depth map duoc normalize de hien thi thanh anh. Gia tri depth dung de so sanh tuong doi giua cac vung trong anh.

Nhom can noi ro:

- Depth map khong nhat thiet la khoang cach met.
- Ket qua chu yeu dung de biet vung nao gan hon, vung nao xa hon.
- Depth map giup bo sung thong tin khong gian cho segmentation.

## Logic ket hop

Sau khi co segmentation mask va depth map, nhom co the tinh hoac nhan xet:

- Depth trung binh cua tung lop.
- Lop nao nam gan camera hon.
- Doi tuong nao can chu y trong canh.
- Vung road phia truoc co lien tuc hay bi vat can che.

## Cach dung AI

Nhom nen dung AI theo huong ho tro hoc va thiet ke, khong dung de lam thay toan bo bai.

Quy trinh dung AI:

- Dua knowledge base truoc: ten de tai, input, output, dataset, model du kien, gioi han pham vi.
- Hoi AI kiem tra cach hieu bai toan.
- Hoi AI gop y feature can co.
- Hoi AI so sanh lua chon model va dataset.
- Hoi AI giai thich logic segmentation, depth va fusion.
- Hoi AI lap checklist implement va test.
- Khong hoi AI theo kieu "lam het project cho em".

## Knowledge base cho AI

Knowledge base can dua cho AI:

```text
Nhom 9 lam de tai scene understanding cho anh duong pho.
Input: anh duong pho.
Output: semantic segmentation mask va depth map.
Dataset tham khao: Cityscapes cho segmentation, KITTI cho depth.
Model tham khao: U-Net cho segmentation, MiDaS cho depth.
Muc tieu: demo phan tich ngu canh giao thong, khong xay dung xe tu hanh hoan chinh.
AI chi ho tro giai thich, gop y logic, chia task va lap checklist test.
```

## Prompt mau nen dung

```text
Nhom em dang lam de tai phan tich ngu canh giao thong bang semantic segmentation va depth estimation. Em da tu xac dinh input la anh duong pho, output la segmentation mask va depth map. Ban kiem tra giup cach hieu nay da dung chua, can gioi han pham vi the nao de de trien khai trong bai tap lon.
```

```text
Voi de tai nay, em du kien dung Cityscapes cho segmentation, KITTI cho depth, U-Net cho segmentation va MiDaS cho depth estimation. Ban phan tich giup vi sao cach chon nay hop ly, diem manh va han che cua tung thanh phan.
```

```text
Em muon giai thich logic ket hop segmentation va depth. Ban giup em dien dat ngan gon: segmentation tra loi pixel thuoc lop gi, depth tra loi vung do gan hay xa, khi ket hop thi he thong hieu ngu canh giao thong tot hon.
```

## Cau tom tat khi thuyet trinh

Nhom em dung AI nhu mot cong cu ho tro phan tich. Truoc khi hoi AI, nhom em dua knowledge base gom yeu cau, input, output, dataset, model va gioi han de tai. AI duoc dung de kiem tra logic, goi y cach trien khai va lap checklist, con viec chot pham vi, viet code, chay demo va danh gia ket qua la do nhom thuc hien.

