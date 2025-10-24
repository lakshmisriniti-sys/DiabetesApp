import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# -----------------------------
# 1️⃣ Ensure folders exist
# -----------------------------
os.makedirs('models', exist_ok=True)  # Folder to save model & scaler

# -----------------------------
# 2️⃣ Load dataset
# -----------------------------
# Adjust the path according to where your CSV is
# If CSV is in DiabetesApp/data/, use 'data/diabetes.csv'
# If CSV is in DiabetesApp/, use 'diabetes.csv'
data = pd.read_csv('data/diabetes.csv')

# -----------------------------
# 3️⃣ Handle missing/zero values
# -----------------------------
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_to_replace] = data[cols_to_replace].replace(0, np.nan)
data[cols_to_replace] = data[cols_to_replace].fillna(data[cols_to_replace].mean())

# -----------------------------
# 4️⃣ Split features and target
# -----------------------------
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# -----------------------------
# 5️⃣ Split into training/testing sets
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 6️⃣ Scale features
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 7️⃣ Train model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 8️⃣ Test model
# -----------------------------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Model trained successfully! Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# 9️⃣ Save model and scaler
# -----------------------------
joblib.dump(model, 'models/diabetes_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("💾 Model and scaler saved successfully in the 'models' folder!")
# -----------------------------
# 6️⃣ Your Random Forest model
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import os
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# -----------------------------
# 7️⃣ Compare with Logistic Regression & SVM
# -----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "RandomForest": rf_model,  # already trained
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

best_model = rf_model
best_accuracy = rf_acc
best_model_name = "RandomForest"

for name, model in models.items():
    if name == "RandomForest":
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:,1]
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_probs = model.predict_proba(X_test_scaled)[:,1]

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_probs))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# -----------------------------
# 8️⃣ Save best model & scaler
# -----------------------------
os.makedirs('models', exist_ok=True)
if best_model_name in ["LogisticRegression", "SVM"]:
    joblib.dump(best_model, 'models/best_diabetes_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
else:
    joblib.dump(best_model, 'models/best_diabetes_model.pkl')

print(f"\n💾 Best model saved: {best_model_name} with accuracy {best_accuracy:.4f}")

