import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("heart.csv")
print("Dataset Loaded!")
print(df.head())

# =========================
# 2. BASIC INFO
# =========================
print("\n==== Missing Values ====")
print(df.isnull().sum())

print("\n==== Statistical Summary ====")
print(df.describe())

# =========================
# 3. CREATE OUTPUT FOLDERS
# =========================
os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =========================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =========================

# 4.1 Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.close()

# 4.2 Distribution of all numeric features
df.hist(figsize=(12, 10), bins=20)
plt.suptitle("Feature Distributions")
plt.savefig("images/feature_distributions.png")
plt.close()

# 4.3 Target Count Plot
plt.figure(figsize=(6,4))
sns.countplot(x=df["target"])
plt.title("Heart Disease Distribution")
plt.savefig("images/target_distribution.png")
plt.close()

# 4.4 Boxplot Example: Cholesterol vs Heart Disease
plt.figure(figsize=(6,4))
sns.boxplot(x="target", y="chol", data=df)
plt.title("Cholesterol vs Heart Disease")
plt.savefig("images/cholesterol_vs_target.png")
plt.close()

# =========================
# 5. TRAIN-TEST SPLIT
# =========================
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# =========================
# 6. FEATURE SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 7. MODEL TRAINING
# =========================
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

accuracies = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc

    print(f"Accuracy: {acc}")
    print(classification_report(y_test, preds))

    # Save confusion matrix
    plt.figure(figsize=(4,3))
    sns.heatmap(confusion_matrix(y_test, preds),
                annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"images/confusion_matrix_{name.replace(' ', '_')}.png")
    plt.close()

    # Save model
    joblib.dump(model, f"models/{name.replace(' ', '_')}.pkl")


# =========================
# 8. ROC CURVE (for Logistic Regression)
# =========================
log_model = models["Logistic Regression"]
pred_prob = log_model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, pred_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("images/roc_curve_logistic.png")
plt.close()

# =========================
# 9. ACCURACY COMPARISON
# =========================
print("\n===== Accuracy Comparison =====")
print(accuracies)
