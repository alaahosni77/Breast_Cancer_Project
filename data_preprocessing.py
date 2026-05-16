# %%
import pandas as pd
from sklearn.datasets import load_breast_cancer

# %%
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
df.info()

# %%
print("Number of missing values per column:")
print(df.isnull().sum())
# %%
print("Number of duplicate rows:", df.duplicated().sum())

# %%
from sklearn.preprocessing import StandardScaler
X = df.drop('target', axis=1)
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print("First 5 rows of the normalized features:")
print(X_scaled_df.head())

# %%
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=15)
X_selected = selector.fit_transform(X_scaled, y)

selected_features = X.columns[selector.get_support()]
print("Selected Features:")
print(selected_features)

# %%
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
X_selected_df['target'] = y.values
X_selected_df.to_csv('processed_breast_cancer.csv', index=False)
print("\nSuccess! Processed data saved to 'processed_breast_cancer.csv'")
