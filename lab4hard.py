import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(r'D:\kpi\AD\automobile\imports-85.data', header=None, na_values='?')
df = df.dropna(axis=1)
df_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
data_np = df_numeric.to_numpy()

# Нормалізація та стандартизація
def normalize_data(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    return (data - data_min) / (data_max - data_min)

def standardize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

normalized_data = normalize_data(data_np)
standardized_data = standardize_data(data_np)

# 3
attribute_index = 2 
plt.figure(figsize=(8, 4))
plt.hist(data_np[:, attribute_index], bins=10, edgecolor='black')
plt.title(f'Гістограма для атрибуту {attribute_index}')
plt.xlabel('Значення')
plt.ylabel('Кількість')
plt.grid(True)
plt.show()

# 4
x_index = 0
y_index = 2
plt.figure(figsize=(6, 4))
plt.scatter(data_np[:, x_index], data_np[:, y_index], alpha=0.7)
plt.title(f'Залежність: Атрибут {y_index} від Атрибуту {x_index}')
plt.xlabel(f"Атрибут {x_index}")
plt.ylabel(f"Атрибут {y_index}")
plt.grid(True)
plt.show()

# 5
pearson_corr, _ = pearsonr(data_np[:, x_index], data_np[:, y_index])
spearman_corr, _ = spearmanr(data_np[:, x_index], data_np[:, y_index])

print(f"Коефіцієнт Пірсона: {pearson_corr:.4f}")
print(f"Коефіцієнт Спірмена: {spearman_corr:.4f}")

# 6
# cat_col_index = 3
# if df[cat_col_index].dtype == 'object':
#     encoded_df = pd.get_dummies(df[cat_col_index], prefix=f'attr{cat_col_index}')
#     print(f"\nOne-Hot Encoding для атрибуту {cat_col_index}:\n")
#     print(encoded_df.head())
# else:
#     print(f"Стовпець {cat_col_index} не є строковим для one-hot encoding.")
cat_col_index = 3
cat_data = df[[cat_col_index]].astype(str).fillna('Missing')  # Переводимо в str для OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)  # або sparse=False залежно від версії sklearn
encoded = encoder.fit_transform(cat_data)
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([f'attr{cat_col_index}']))
print(f"\nOne-Hot Encoding для атрибуту {cat_col_index}:\n")
print(encoded_df.head())

# 7

sample_df = df.select_dtypes(include=[np.number]).iloc[:, :5] 
sns.pairplot(sample_df.dropna())
plt.suptitle("Pairplot для перших 5 числових атрибутів", y=1.02)
plt.show()
