import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV vào DataFrame
df = pd.read_csv('./data_train/Breathing.csv')

Column1 = 'aX'
Column2 = 'aY'
Column3 = 'aZ'

# Column1 = 'gX'
# Column2 = 'gY'
# Column3 = 'gZ'


# Vẽ biểu đồ cho ba cột
plt.plot(df[Column1], label=Column1)
plt.plot(df[Column2], label=Column2)
plt.plot(df[Column3], label=Column3)
plt.title('Biểu đồ cho ba cột')
plt.xlabel('Chỉ số dòng')
plt.ylabel('Giá trị')
plt.legend()
plt.show()
