# -------------------------------
# Rain Prediction Project
# -------------------------------

# 1️⃣ Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# 2️⃣ Load Dataset
df = pd.read_csv("weather.csv")

print("Column Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

# 3️⃣ Create Rain Column (1 if Rain appears in Weather column)
df['Rain'] = df['Weather'].apply(lambda x: 1 if 'Rain' in x else 0)

# 4️⃣ Select Features (Inputs)
X = df[['Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Press_kPa']]

# 5️⃣ Target (Output)
y = df['Rain']

# 6️⃣ Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7️⃣ Train Model

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8️⃣ Predictions
y_pred = model.predict(X_test)

# 9️⃣ Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# 🔟 Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt

# Feature Importance
importance = model.feature_importances_
features = X.columns

plt.bar(features, importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()
