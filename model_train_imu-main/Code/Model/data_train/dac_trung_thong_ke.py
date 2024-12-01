import pandas as pd
import numpy as np

# Đọc dữ liệu từ file CSV
ngungtho_data = pd.read_csv('ngungtho_data_cleaned.csv')
tho_data = pd.read_csv('tho_data_cleaned.csv')

# Khám phá dữ liệu
print(ngungtho_data.head())
print(ngungtho_data.describe())
print(tho_data.head())
print(tho_data.describe())

# Tiền xử lý dữ liệu (ví dụ: xử lý missing values)
# ngungtho_data = ngungtho_data.dropna()
# tho_data = tho_data.dropna()

# Trích xuất đặc trưng
def extract_features(data):
    features = {}
    features['mean'] = np.mean(data['aZ'])
    features['std'] = np.std(data['aZ'])
    features['min'] = np.min(data['aZ'])
    features['max'] = np.max(data['aZ'])
    features['energy'] = np.sum(data['aZ']**2)
    features['entropy'] = -np.sum(data['aZ'] * np.log2(data['aZ'] + 1e-12))
    # Add more features as needed
    return features

# Trích xuất đặc trưng từ dữ liệu ngưng thở
ngungtho_features = extract_features(ngungtho_data)

# Trích xuất đặc trưng từ dữ liệu thở bình thường
tho_features = extract_features(tho_data)

# Tạo DataFrame chứa các đặc trưng
features_df = pd.DataFrame([ngungtho_features, tho_features], index=['ngungtho', 'tho'])

print(features_df)
