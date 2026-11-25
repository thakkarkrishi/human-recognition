# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load Dataset
df = pd.read_csv("test.csv")

# 3. EDA
print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nActivity Distribution:")
print(df["Activity"].value_counts())

# Activity Distribution Graph
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Activity")
plt.title("Activity Distribution")
plt.xticks(rotation=30)
plt.show()

# 4. Separate Features and Target
X = df.drop(columns=["Activity", "subject"], errors="ignore")
y = df["Activity"]

print("\nTotal Features:", X.shape[1])

# 5. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Standardization
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. PCA
pca = PCA(n_components=0.95)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nOriginal Features:", X.shape[1])
print("PCA Features:", X_train_pca.shape[1])

print("Variance Retained:",
      sum(pca.explained_variance_ratio_) * 100)

# PCA Graph
plt.figure(figsize=(8, 5))
plt.plot(
    np.cumsum(pca.explained_variance_ratio_)
)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid()
plt.show()

# 8. Train Model
model = LogisticRegression(max_iter=3000)

model.fit(X_train_pca, y_train)

# 9. Prediction
y_pred = model.predict(X_test_pca)

# 10. Model Evaluation
print("\n========== MODEL EVALUATION ==========")

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 11. Save Model, Scaler and PCA
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

joblib.dump(X.columns.tolist(), "columns.pkl")

print("\nModel, Scaler and PCA Saved Successfully!")