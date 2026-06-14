import os
import matplotlib.pyplot as plt
import numpy as np

# Cấu hình phông chữ hiển thị tiếng Việt
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# Dữ liệu ma trận nhầm lẫn (tỷ lệ phần trăm đã chuẩn hóa)
classes = ['Bầu trời (Sky)', 'Mặt đường (Road)', 'Vỉa hè (Sidewalk)']
cm = np.array([
    [0.97, 0.00, 0.03],  # Thực tế là Bầu trời (Actual Sky)
    [0.00, 0.88, 0.12],  # Thực tế là Mặt đường (Actual Road)
    [0.01, 0.17, 0.82]   # Thực tế là Vỉa hè (Actual Sidewalk)
])

fig, ax = plt.subplots(figsize=(8, 6.5))

# Vẽ Heatmap sử dụng colormap màu xanh (Blues)
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)

# Thêm thanh màu sắc bên cạnh (Colorbar)
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel('Tỷ lệ trùng khớp', rotation=-90, va="bottom", fontsize=11)

# Đặt nhãn cho các trục
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, fontsize=11)
ax.set_yticklabels(classes, fontsize=11)

# Xoay nhãn trục X để hiển thị đẹp hơn
plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

# Điền các giá trị phần trăm vào từng ô
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        text_color = "white" if val > thresh else "black"
        ax.text(j, i, f"{val*100:.1f}%",
                ha="center", va="center",
                color=text_color, fontsize=12, weight='bold')

# Tiêu đề và nhãn
ax.set_title("Ma trận nhầm lẫn (Confusion Matrix Heatmap)", fontsize=14, weight='bold', pad=20)
ax.set_xlabel('Lớp dự đoán (Predicted Class)', fontsize=12, labelpad=10)
ax.set_ylabel('Lớp thực tế (Actual Class)', fontsize=12, labelpad=10)

plt.tight_layout()

# Đường dẫn lưu file kết quả trong thư mục dự án
output_path = "confusion_matrix_heatmap.png"
plt.savefig(output_path, dpi=300)
print(f"Đã vẽ và lưu biểu đồ heatmap thành công tại: {os.path.abspath(output_path)}")
