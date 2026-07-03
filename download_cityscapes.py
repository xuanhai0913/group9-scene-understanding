import os
import sys
import requests
from getpass import getpass

def download_file(session, package_id, destination_path):
    download_url = f"https://www.cityscapes-dataset.com/file-handling/?packageID={package_id}"
    
    print(f"[INFO] Bắt đầu tải package ID {package_id}...")
    
    # Gửi yêu cầu tải file
    response = session.get(download_url, stream=True)
    if response.status_code != 200:
        print(f"[ERROR] Không thể tải file. HTTP Status Code: {response.status_code}")
        return False
        
    # Lấy dung lượng file nếu có
    total_size = int(response.headers.get('content-length', 0))
    
    # Ghi file ra ổ đĩa
    with open(destination_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    sys.stdout.write(f"\r[Tải xuống] {percent:.2f}% ({downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB)")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\r[Tải xuống] Đã tải: {downloaded/(1024*1024):.1f}MB")
                    sys.stdout.flush()
    print(f"\n[SUCCESS] Đã lưu thành công tại: {destination_path}")
    return True

def main():
    print("="*60)
    print(" CÔNG CỤ TẢI TỰ ĐỘNG BỘ DỮ LIỆU CITYSCAPES TRÊN GOOGLE COLAB")
    print("="*60)
    
    # Nhập tài khoản đăng nhập Cityscapes
    username = input("Nhập tên đăng nhập (User Name hoặc Email) của Cityscapes: ")
    password = getpass("Nhập mật khẩu Cityscapes (ký tự nhập sẽ ẩn đi): ")
    
    if not username or not password:
        print("[ERROR] Tài khoản hoặc mật khẩu không được để trống!")
        return

    # Đường dẫn thư mục lưu trữ trên Colab
    dest_dir = "./data/cityscapes"
    os.makedirs(dest_dir, exist_ok=True)
    
    # Khởi tạo session để lưu cookie đăng nhập
    session = requests.Session()
    login_url = "https://www.cityscapes-dataset.com/login/"
    
    # Dữ liệu POST để đăng nhập
    login_data = {
        "username": username,
        "password": password,
        "submit": "Login"
    }
    
    print("\n[INFO] Đang thực hiện đăng nhập vào hệ thống Cityscapes...")
    login_response = session.post(login_url, data=login_data)
    
    # Kiểm tra xem có đăng nhập thành công hay không bằng cách check cookie
    if not any(cookie.name == 'cityscapes_session' for cookie in session.cookies):
        print("[ERROR] Đăng nhập thất bại! Vui lòng kiểm tra lại tài khoản, mật khẩu hoặc xác nhận kích hoạt email của bạn.")
        return
        
    print("[SUCCESS] Đăng nhập thành công!")
    
    # 1. Tải gtFine_trainvaltest.zip (Package ID = 1)
    gt_path = os.path.join(dest_dir, "gtFine_trainvaltest.zip")
    download_file(session, 1, gt_path)
    
    # 2. Tải leftImg8bit_trainvaltest.zip (Package ID = 3)
    img_path = os.path.join(dest_dir, "leftImg8bit_trainvaltest.zip")
    download_file(session, 3, img_path)
    
    print("\n" + "="*60)
    print(" HOÀN THÀNH TẢI BỘ DỮ LIỆU CITYSCAPES!")
    print("="*60)

if __name__ == "__main__":
    main()
