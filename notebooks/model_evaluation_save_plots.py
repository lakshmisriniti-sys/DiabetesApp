# ==============================
# Diabetes Model Evaluation + Plots
# ==============================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score,
    roc_curve, classification_report, confusion_matrix
)

# ==============================
# File Paths (adjusted for your setup)
# ==============================
data_path = "data/diabetes.csv"          # ✅ Direct path
save_dir = "plots"                       # ✅ Where plots will be saved
model_save_path = "models/diabetes_model.pkl"

# Ensure folders exist
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# ==============================
# Load Dataset
# ==============================
data = pd.read_csv(data_path)

print("Class distribution:")
print(data["Outcome"].value_counts(), "\n")

# Split features and target
X = data.drop(columns=["Outcome"])
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# Model Training with Grid Search
# ==============================
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42, class_weight="balanced")
grid_search = GridSearchCV(
    rf, param_grid, scoring="roc_auc", cv=5, n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# ==============================
# Model Evaluation
# ==============================
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Recall: {recall:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(best_model, model_save_path)

# ==============================
# Feature Importance
# ==============================
importances = best_model.feature_importances_
feature_names = X.columns
indices = importances.argsort()

plt.figure(figsize=(8, 6))
sns.barplot(x=importances[indices], y=feature_names[indices], palette="viridis", legend=False)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "feature_importance.png"))
plt.close()

# ==============================
# Confusion Matrix
# ==============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
plt.close()

# ==============================
# ROC Curve
# ==============================
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "roc_curve.png"))
plt.close()

# ==============================
# Save Report
# ==============================
report_path = os.path.join(save_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("Best Parameters:\n")
    f.write(str(grid_search.best_params_) + "\n\n")
    f.write(f"Accuracy: {accuracy:.3f}\n")
    f.write(f"Recall: {recall:.3f}\n")
    f.write(f"ROC-AUC: {roc_auc:.3f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))

print(f"\n✅ All plots and reports saved in: {os.path.abspath(save_dir)}")
