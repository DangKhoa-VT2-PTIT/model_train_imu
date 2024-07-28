import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV vào DataFrame
df = pd.read_csv('./data_train/DiBo.csv')

# Chọn cột để vẽ biểu đồ (ví dụ: cột 'Column_name')
column_name = 'gZ'

# Vẽ biểu đồ
plt.plot(df[column_name])
plt.title(f'Biểu đồ cho cột {column_name}')
plt.xlabel('Chỉ số dòng')
plt.ylabel('Giá trị')
plt.show()
