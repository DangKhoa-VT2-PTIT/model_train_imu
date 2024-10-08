import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
ngungtho_data = pd.read_csv('ngungtho_data_cleaned.csv')
tho_data = pd.read_csv('tho_data_cleaned.csv')

# Kiểm tra các cột có trong dữ liệu
print("Columns in ngungtho_data:", ngungtho_data.columns)
print("Columns in tho_data:", tho_data.columns)

# Thay thế 'accel_z' bằng tên cột chứa dữ liệu bạn muốn phân tích
ngungtho_z = ngungtho_data['aZ'].to_numpy()
tho_z = tho_data['aZ'].to_numpy()

# Đảm bảo dữ liệu là dạng float
ngungtho_z = ngungtho_z.astype(float)
tho_z = tho_z.astype(float)

# Tần số mẫu (Hz), bạn cần thay thế giá trị này bằng tần số thực tế của dữ liệu bạn
sampling_freq = 100  # ví dụ: 100 Hz

# Thực hiện phân tích Fourier
ngungtho_fft = fft(ngungtho_z)
tho_fft = fft(tho_z)

# Tính tần số tương ứng
ngungtho_freqs = fftfreq(len(ngungtho_z), 1/sampling_freq)
tho_freqs = fftfreq(len(tho_z), 1/sampling_freq)

# Giới hạn phổ công suất đến 2 Hz
ngungtho_mask = (ngungtho_freqs >= 0) & (ngungtho_freqs <= 2)
tho_mask = (tho_freqs >= 0) & (tho_freqs <= 2)

ngungtho_psd = np.abs(ngungtho_fft[ngungtho_mask]) ** 2
tho_psd = np.abs(tho_fft[tho_mask]) ** 2

# Tần số tương ứng với phổ công suất đã giới hạn
ngungtho_freqs = ngungtho_freqs[ngungtho_mask]
tho_freqs = tho_freqs[tho_mask]

# Vẽ phổ công suất
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(ngungtho_freqs, ngungtho_psd)
plt.title('Phổ công suất - Ngưng thở')
plt.xlabel('Tần số (Hz)')
plt.ylabel('Công suất')

plt.subplot(1, 2, 2)
plt.plot(tho_freqs, tho_psd)
plt.title('Phổ công suất - Thở bình thường')
plt.xlabel('Tần số (Hz)')
plt.ylabel('Công suất')

plt.tight_layout()
plt.show()
