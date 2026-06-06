# Features

## Feature chinh

He thong gom cac tinh nang chinh sau.

## Doc anh dau vao

He thong doc mot anh duong pho tu thu muc du lieu hoac tu file upload. Anh dau vao duoc resize ve kich thuoc phu hop de dua vao model. Buoc nay giup dam bao anh co dung dinh dang va kich thuoc ma model can.

## Semantic segmentation

Tinh nang nay tao segmentation mask cho anh dau vao. Moi pixel trong anh duoc gan vao mot lop ngu nghia.

Cac lop uu tien:

- road
- car
- sky
- sidewalk
- person
- building
- vegetation

Ket qua hien thi:

- Mask mau rieng cho tung lop.
- Overlay mask len anh goc de de quan sat.

## Depth estimation

Tinh nang nay tao depth map tu anh dau vao. Depth map cho thay vung nao gan va vung nao xa camera.

Ket qua hien thi:

- Anh depth map dang grayscale hoac colormap.
- Co the dung mau sang/toi de bieu dien muc do gan xa.

Luu y: voi MiDaS, depth thuong la relative depth, tuc la so sanh gan xa tuong doi chu khong phai khoang cach met chinh xac.

## Fusion segmentation va depth

Sau khi co segmentation va depth, he thong ket hop hai ket qua de rut ra nhan xet ngu canh.

Vi du:

- Lop road nam o phia truoc va co do sau thay doi theo phoi canh.
- Xe gan camera co depth khac xe o xa.
- Sky thuong la vung xa.
- Sidewalk nam hai ben duong va co the tach khoi road.

## Visual report

He thong can tao anh tong hop de demo:

- Anh goc.
- Segmentation mask.
- Overlay segmentation.
- Depth map.
- Bang nhan xet ngan.

## Feature phu neu con thoi gian

Co the them cac tinh nang mo rong:

- Tinh depth trung binh theo tung lop segmentation.
- Lam noi bat cac vung gan camera.
- So sanh depth cua xe gan va xe xa.
- Chay demo tren nhieu anh duong pho.
- Tao notebook de show code va output.

## Cau tom tat khi thuyet trinh

Feature cua nhom em tap trung vao hai ket qua truc quan: phan vung ngu nghia va ban do do sau. Phan vung giup biet tung vung la gi, con do sau giup biet vung do gan hay xa. Khi ket hop lai, he thong co the phan tich ngu canh giao thong tot hon.

