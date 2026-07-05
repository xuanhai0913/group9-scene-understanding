import os
import sys
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

import matplotlib.pyplot as plt
import numpy as np

# Cấu hình phông chữ hiển thị tiếng Việt
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# Dữ liệu ma trận nhầm lẫn cho 3 lớp chính (Độ chính xác dạng chuẩn hóa)
classes = ['Nền (Background)', 'Đối tượng (Object)']

# 1. Lớp Đường (flat)
cm_road = np.array([
    [0.8716, 0.1284],  # Actual Background -> Predicted Neg, Pos
    [0.3015, 0.6985]   # Actual Road -> Predicted Neg, Pos
])

# 2. Lớp Bầu trời (sky)
cm_sky = np.array([
    [0.9942, 0.0058],  # Actual Background -> Predicted Neg, Pos
    [0.3137, 0.6863]   # Actual Sky -> Predicted Neg, Pos
])

# 3. Lớp Xe cộ (vehicle)
cm_vehicle = np.array([
    [0.9591, 0.0409],  # Actual Background -> Predicted Neg, Pos
    [0.1994, 0.8006]   # Actual Vehicle -> Predicted Neg, Pos
])

cms = [cm_road, cm_sky, cm_vehicle]
titles = [
    "Ma trận nhầm lẫn lớp Đường (Road)",
    "Ma trận nhầm lẫn lớp Bầu trời (Sky)",
    "Ma trận nhầm lẫn lớp Xe cộ (Vehicle)"
]
class_labels = [
    ['Nền (BG)', 'Mặt đường (Road)'],
    ['Nền (BG)', 'Bầu trời (Sky)'],
    ['Nền (BG)', 'Xe cộ (Vehicle)']
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for idx, ax in enumerate(axes):
    cm = cms[idx]
    labels = class_labels[idx]
    
    # Vẽ Heatmap sử dụng colormap màu xanh (Blues)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    
    # Đặt nhãn cho các trục
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    
    # Xoay nhãn trục X
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")
    
    # Điền các giá trị phần trăm vào từng ô
    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text_color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val*100:.2f}%",
                    ha="center", va="center",
                    color=text_color, fontsize=11, weight='bold')
            
    ax.set_title(titles[idx], fontsize=11, weight='bold', pad=15)
    ax.set_xlabel('Lớp dự đoán (Predicted Class)', fontsize=10, labelpad=8)
    if idx == 0:
        ax.set_ylabel('Lớp thực tế (Actual Class)', fontsize=10, labelpad=8)

# Thêm thanh màu sắc bên phải ngoài cùng
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.set_ylabel('Tỷ lệ trùng khớp', rotation=-90, va="bottom", fontsize=10)

plt.suptitle("MA TRẬN NHẦM LẪN CHI TIẾT 3 LỚP CHÍNH (EPOCH 50)", fontsize=14, weight='bold', y=0.98)

# Đường dẫn lưu file kết quả trong thư mục dự án
output_path = "confusion_matrix_heatmap.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Đã vẽ và lưu biểu đồ heatmap thành công tại: {os.path.abspath(output_path)}")
