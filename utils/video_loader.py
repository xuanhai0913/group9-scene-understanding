import os
import sys
import glob
import time
import threading
import queue
import cv2

def get_video_path_interactive(project_root):
    """Opens a CLI menu or a TK File Dialog to select a traffic sample video."""
    search_dir = os.path.join(project_root, 'data', 'sample_videos')
    
    video_files = sorted(
        glob.glob(os.path.join(search_dir, "*.mp4")) + \
        glob.glob(os.path.join(search_dir, "*.mov")) + \
        glob.glob(os.path.join(search_dir, "*.avi")) + \
        glob.glob(os.path.join(search_dir, "*.mkv"))
    )
    
    if not video_files:
        video_files = sorted(
            glob.glob("*.mp4") + glob.glob("*.mov") + glob.glob("*.avi") + glob.glob("*.mkv")
        )
        
    print("\n" + "="*70)
    print("                 DANH SÁCH VIDEO CÓ SẴN ĐỂ CHẠY HỆ THỐNG")
    print("="*70)
    for idx, path in enumerate(video_files):
        filename = os.path.basename(path)
        desc = ""
        if "ho-chi-minh" in filename.lower():
            desc = " (Kẹt xe TP.HCM - Video 1)"
        elif "hanoi" in filename.lower():
            desc = " (Xa lộ Xa Lộ Hà Nội - Video 2)"
        elif "video3lightneed" in filename.lower():
            desc = " (Đèn tín hiệu giao thông - Video 3)"
        print(f"  [{idx + 1}] {filename}{desc}")
        
    gui_option_idx = len(video_files) + 1
    print(f"  [{gui_option_idx}] Chọn video khác từ File Explorer (Giao diện đồ họa)")
    print("="*70)
    
    while True:
        try:
            choice = input(f"Nhập số để chọn (1-{gui_option_idx}) hoặc kéo thả file video vào đây: ").strip()
            # Remove quotes if drag and drop on Windows
            choice = choice.strip('"').strip("'")
            
            if not choice:
                continue
                
            if choice.isdigit():
                val = int(choice)
                if 1 <= val <= len(video_files):
                    selected = video_files[val - 1]
                    print(f"[INFO] Đã chọn video: {selected}")
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
                        print("[INFO] Đang mở hộp thoại chọn file video...")
                        file_path = filedialog.askopenfilename(
                            title="Chọn video giao thông cần phân tích",
                            initialdir=initial_dir,
                            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")]
                        )
                        root.destroy()
                        if file_path:
                            print(f"[INFO] Đã chọn video qua hộp thoại: {file_path}")
                            return file_path
                        else:
                            print("[INFO] Đã hủy chọn bằng File Explorer. Vui lòng chọn lại từ menu.")
                    except Exception as e:
                        print(f"[WARNING] Không thể mở hộp thoại GUI (Lỗi: {e}). Vui lòng chọn số từ danh sách.")
                else:
                    print(f"[WARNING] Lựa chọn không hợp lệ. Vui lòng nhập số từ 1 đến {gui_option_idx}.")
            elif os.path.exists(choice):
                print(f"[INFO] Đã chọn video qua kéo thả: {choice}")
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
    def __init__(self, cap, queue_maxsize=3, skip_frames=1):
        super().__init__()
        self.cap = cap
        self.queue = queue.Queue(maxsize=queue_maxsize)
        self.skip_frames = max(1, skip_frames)
        self.stopped = False
        self.daemon = True

    def run(self):
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
