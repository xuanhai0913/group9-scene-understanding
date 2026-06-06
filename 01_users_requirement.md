# User's Requirement

## Yeu cau bai toan

Nhom 9 chon de tai phan tich ngu canh giao thong dua tren anh duong pho. Input cua he thong la mot anh duong pho, output gom hai phan chinh: semantic segmentation va depth map.

Semantic segmentation dung de phan loai tung pixel trong anh. Moi pixel se duoc gan vao mot lop ngu nghia nhu road, car, sky, sidewalk, building, person hoac vegetation. Ket qua nay giup he thong biet tung vung trong anh dang dai dien cho doi tuong nao.

Depth estimation dung de uoc luong khoang cach tu camera den cac vung trong anh. Output la depth map, trong do moi diem anh the hien muc do gan hoac xa tuong doi. Trong de tai nay, depth khong nhat thiet phai la khoang cach met tuyet doi, ma co the dung de nhan xet vung nao gan hon, vung nao xa hon.

## Input

Input du kien:

- Anh duong pho chup tu goc nhin xe hoac camera giao thong.
- Anh co cac thanh phan quen thuoc nhu mat duong, xe, bau troi, via he, nguoi di bo, cay xanh, nha cua.
- Anh co the lay tu Cityscapes, KITTI hoac anh duong pho tu internet de demo.

## Output

Output mong muon:

- Anh segmentation mask, moi lop co mot mau rieng.
- Anh overlay segmentation len anh goc.
- Depth map bieu dien vung gan/xa.
- Hinh tong hop gom anh goc, segmentation, depth va nhan xet ngan.

## Muc tieu cua nhom

Muc tieu khong phai xay dung xe tu hanh hoan chinh. Muc tieu la minh hoa cach may tinh hieu canh giao thong tu anh: vung nao la duong, dau la xe, dau la bau troi va cac doi tuong do gan hay xa camera.

## Gioi han pham vi

De tai nen gioi han o muc inference va demo tren anh mau. Neu may tinh khong du manh de train model lon, nhom co the dung model da huan luyen san. Neu co train, chi train hoac fine-tune tren mot tap nho de minh hoa quy trinh.

## Cau tom tat khi thuyet trinh

De tai cua nhom em nhan anh duong pho lam dau vao, sau do phan vung tung pixel theo doi tuong va uoc luong do sau tuong doi cua tung vung. Nhu vay he thong khong chi nhan biet anh co gi, ma con hieu duoc cau truc khong gian gan xa trong canh giao thong.

