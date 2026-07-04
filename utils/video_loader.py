import os
import sys
import glob
import time
import threading
import queue
import cv2

def get_video_path_interactive(project_root):
    """Opens a CLI menu or a TK File Dialog to select a traffic sample video or image."""
    search_dir = os.path.join(project_root, 'data', 'sample_videos')
    
    # Kênh tìm kiếm hỗ trợ cả video và ảnh tĩnh
    extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]
    files = []
    if os.path.exists(search_dir):
        for ext in extensions:
            files.extend(glob.glob(os.path.join(search_dir, ext)))
    files = sorted(list(set(files)))
    
    if not files:
        for ext in extensions:
            files.extend(glob.glob(ext))
        files = sorted(list(set(files)))
        
    print("\n" + "="*70)
    print("                 DANH SÁCH VIDEO / ẢNH CÓ SẴN ĐỂ CHẠY HỆ THỐNG")
    print("="*70)
    for idx, path in enumerate(files):
        filename = os.path.basename(path)
        desc = ""
        if "ho-chi-minh" in filename.lower():
            desc = " (Kẹt xe TP.HCM - Video 1)"
        elif "hanoi" in filename.lower():
            desc = " (Xa lộ Xa Lộ Hà Nội - Video 2)"
        elif "video3lightneed" in filename.lower():
            desc = " (Đèn tín hiệu giao thông - Video 3)"
        elif any(filename.lower().endswith(ext.replace("*", "")) for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]):
            desc = " (Ảnh tĩnh)"
        print(f"  [{idx + 1}] {filename}{desc}")
        
    gui_option_idx = len(files) + 1
    print(f"  [{gui_option_idx}] Chọn file khác từ File Explorer (Giao diện đồ họa)")
    print("="*70)
    
    while True:
        try:
            choice = input(f"Nhập số để chọn (1-{gui_option_idx}) hoặc kéo thả file video/ảnh vào đây: ").strip()
            # Remove quotes if drag and drop on Windows
            choice = choice.strip('"').strip("'")
            
            if not choice:
                continue
                
            if choice.isdigit():
                val = int(choice)
                if 1 <= val <= len(files):
                    selected = files[val - 1]
                    print(f"[INFO] Đã chọn file: {selected}")
                    return selected
                elif val == gui_option_idx:
                    # Open Tkinter file dialog
                    try:
                        import tkinter as tk
                        from tkinter import filedialog
                        
                        root = tk.Tk()
                        root.withdraw()
                        root.attributes("-topmost", True)
                        
                        initial_dir = search_dir if os.path.exists(search_dir) else os.getcwd()
                        print("[INFO] Đang mở hộp thoại chọn file...")
                        file_path = filedialog.askopenfilename(
                            title="Chọn video hoặc ảnh giao thông cần phân tích",
                            initialdir=initial_dir,
                            filetypes=[
                                ("Video/Image files", "*.mp4 *.mov *.avi *.mkv *.png *.jpg *.jpeg *.webp *.bmp"),
                                ("All files", "*.*")
                            ]
                        )
                        root.destroy()
                        if file_path:
                            print(f"[INFO] Đã chọn qua hộp thoại: {file_path}")
                            return file_path
                        else:
                            print("[INFO] Đã hủy chọn bằng File Explorer. Vui lòng chọn lại từ menu.")
                    except Exception as e:
                        print(f"[WARNING] Không thể mở hộp thoại GUI (Lỗi: {e}). Vui lòng chọn số từ danh sách.")
                else:
                    print(f"[WARNING] Lựa chọn không hợp lệ. Vui lòng nhập số từ 1 đến {gui_option_idx}.")
            elif os.path.exists(choice):
                print(f"[INFO] Đã chọn qua kéo thả: {choice}")
                return choice
            else:
                print(f"[WARNING] Đường dẫn file hoặc lựa chọn không hợp lệ. Vui lòng thử lại.")
        except (KeyboardInterrupt, SystemExit):
            print("\n[INFO] Đã thoát chương trình.")
            sys.exit(0)
        except Exception as e:
            print(f"[WARNING] Có lỗi xảy ra: {e}. Vui lòng thử lại.")

class VideoReaderThread(threading.Thread):
    """Background thread that continuously reads and pre-buffers frames from VideoCapture."""
    def __init__(self, cap, queue_maxsize=3, skip_frames=1, is_image_input=False, video_path=""):
        super().__init__()
        self.cap = cap
        self.queue = queue.Queue(maxsize=queue_maxsize)
        self.skip_frames = max(1, skip_frames)
        self.is_image_input = is_image_input
        self.video_path = video_path
        self.stopped = False
        self.daemon = True

    def run(self):
        if self.is_image_input:
            frame = cv2.imread(self.video_path)
            if frame is not None:
                self.queue.put(frame)
            self.stopped = True
            return

        frame_idx = 0
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                if frame_idx % self.skip_frames == 0:
                    self.queue.put(frame)
                frame_idx += 1
            else:
                time.sleep(0.005)

    def stop(self):
        self.stopped = True
