import os
import sys
import requests

# Thiet lap ma hoa UTF-8 cho stdout de tranh loi Unicode tren Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Dinh nghia headers gia lap trinh duyet de tranh bi chan bot
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def download_file(session, package_id, destination_path):
    download_url = f"https://www.cityscapes-dataset.com/file-handling/?packageID={package_id}"
    
    print(f"[INFO] Bat dau tai package ID {package_id}...")
    sys.stdout.flush()
    
    response = session.get(download_url, headers=HEADERS, stream=True)
    if response.status_code != 200:
        print(f"[ERROR] Khong the tai file. HTTP Status Code: {response.status_code}")
        sys.stdout.flush()
        return False
        
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    sys.stdout.write(f"\r[Tai xuong] {percent:.2f}% ({downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB)")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\r[Tai xuong] Da tai: {downloaded/(1024*1024):.1f}MB")
                    sys.stdout.flush()
    print(f"\n[SUCCESS] Da luu thanh cong tai: {destination_path}")
    sys.stdout.flush()
    return True

def main():
    print("="*60)
    print(" CONG CU TAI TU DONG BO DU LIEU CITYSCAPES")
    print("="*60)
    sys.stdout.flush()
    
    username = input("Nhap ten dang nhap (User Name hoac Email) cua Cityscapes: ")
    password = input("Nhap mat khau Cityscapes (mat khau se hien thi khi nhap tren terminal nay): ")
    
    if not username or not password:
        print("[ERROR] Tai khoan hoac mat khau khong duoc de trong!")
        sys.stdout.flush()
        return

    dest_dir = "./data/cityscapes"
    os.makedirs(dest_dir, exist_ok=True)
    
    session = requests.Session()
    login_url = "https://www.cityscapes-dataset.com/login/"
    
    # Buoc 1: Goi GET truoc de khoi tao session va nhan cookie ban dau
    print("[INFO] Khoi tao phien lam viec...")
    sys.stdout.flush()
    session.get(login_url, headers=HEADERS)
    
    login_data = {
        "username": username,
        "password": password,
        "submit": "Login"
    }
    
    # Buoc 2: Gui POST de dang nhap
    print("[INFO] Dang thuc hien dang nhap vao he thong Cityscapes...")
    sys.stdout.flush()
    login_response = session.post(login_url, data=login_data, headers=HEADERS)
    
    # Buoc 3: Kiem tra dang nhap thanh cong bang cach tim chu "Logout" hoac "logout" trong HTML tra ve
    if "logout" not in login_response.text.lower():
        print("[ERROR] Dang nhap that bai! Vui long kiem tra lai tai khoan, mat khau hoac xac nhan kich hoat email cua ban.")
        sys.stdout.flush()
        return
        
    print("[SUCCESS] Dang nhap thanh cong!")
    sys.stdout.flush()
    
    # 1. Tai gtFine_trainvaltest.zip (Package ID = 1)
    gt_path = os.path.join(dest_dir, "gtFine_trainvaltest.zip")
    download_file(session, 1, gt_path)
    
    # 2. Tai leftImg8bit_trainvaltest.zip (Package ID = 3)
    img_path = os.path.join(dest_dir, "leftImg8bit_trainvaltest.zip")
    download_file(session, 3, img_path)
    
    print("\n[INFO] Bat dau giai nen gtFine_trainvaltest.zip...")
    sys.stdout.flush()
    import zipfile
    with zipfile.ZipFile(gt_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
        
    print("[INFO] Bat dau giai nen leftImg8bit_trainvaltest.zip (Co the mat vai phut)...")
    sys.stdout.flush()
    with zipfile.ZipFile(img_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
        
    print("\n" + "="*60)
    print(" HOAN THANH TAI VA GIAI NEN BO DU LIEU CITYSCAPES!")
    print("="*60)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
