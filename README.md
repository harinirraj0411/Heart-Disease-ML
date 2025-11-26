# ❤️ Heart Disease Prediction using Machine Learning

This project builds a **Heart Disease Classification Model** using multiple machine learning algorithms.  
It includes **full EDA, data preprocessing, model training, evaluation, visualizations, and saved models**.

The goal is to predict whether a patient is likely to have heart disease based on clinical features such as age, cholesterol level, blood pressure, chest pain type, etc.

---

## 🚀 **Project Highlights**

- ✔ Complete Exploratory Data Analysis (EDA)
- ✔ Data Cleaning & Preprocessing
- ✔ Feature Scaling with StandardScaler
- ✔ Trained Multiple ML models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
- ✔ Confusion Matrix Visualization for each model
- ✔ ROC Curve for Logistic Regression
- ✔ Automated saving of plots & trained models
- ✔ Organized folder structure for GitHub

---
Heart-Disease-ML-Classification/
│── heart_disease.py
│── README.md
│── .gitignore
│
├── models/ # Saved ML models (ignored in GitHub)
│ ├── Logistic_Regression.pkl
│ ├── Random_Forest.pkl
│ └── SVM.pkl
│
├── images/ # All visualizations (ignored)
├── correlation_heatmap.png
├── feature_distributions.png
├── target_distribution.png
├── cholesterol_vs_target.png
├── confusion_matrix_Logistic_Regression.png
├── confusion_matrix_Random_Forest.png
├── confusion_matrix_SVM.png
└── roc_curve_logistic.png



---

## 📊 **Dataset Information**

The dataset used is the popular **Heart Disease UCI dataset** containing:

- 1025 rows  
- 14 features  
- Target variable:  
  - `1` → Heart Disease  
  - `0` → No Heart Disease  

### **Features include:**
- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure  
- Serum Cholesterol  
- Fasting Blood Sugar  
- Resting ECG  
- Maximum Heart Rate Achieved  
- Exercise Induced Angina  
- Oldpeak  
- Slope  
- CA  
- Thal  

---

## 🔍 **Exploratory Data Analysis (EDA)**

The script automatically generates:

### 📌 **1. Correlation Heatmap**  
Shows feature relationships.

### 📌 **2. Distributions of All Features**  
To understand data spread & variations.

### 📌 **3. Target Variable Distribution**  
Identifies class imbalance.

### 📌 **4. Boxplot**  
Example: Cholesterol levels vs Heart Disease.

### 📌 **5. Pairplots (optional)**  
Visual relationships between features.

All plots are saved inside the **images/** folder.

---

## 🤖 **Machine Learning Models Used**

Three ML algorithms were trained to compare performance:

### **1️⃣ Logistic Regression**
- Accuracy: ~79%

### **2️⃣ Random Forest Classifier**
- ⭐ **Best Model**
- Accuracy: ~99%

### **3️⃣ Support Vector Machine (SVM)**
- Accuracy: ~89%

---

## 📈 **Accuracy Comparison**

| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | 0.79     |
| Random Forest        | ⭐ 0.99  |
| SVM                  | 0.89     |

---

## 📉 **Evaluation Metrics**

For each model, the script automatically generates:

- Confusion Matrix
- Precision, Recall, F1-score
- Support for each class
- ROC Curve (Logistic Regression)

---

## 🧠 **Technologies Used**

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-Learn  
- Joblib  

---

## ▶️ **How to Run the Project**

### **1. Install dependencies**
pip install -r requirements.txt


Or install manually:



pip install pandas numpy matplotlib seaborn scikit-learn joblib


---

### **2. Place your dataset**

Add the dataset file named:



heart.csv


in the project folder.

---

### **3. Run script**



python heart_disease.py


---

### **4. Check outputs**

📁 **images/** → all plots  
📁 **models/** → trained model files  
Console → accuracy, reports, metrics  

---

## ✨ **Future Improvements**

- Add hyperparameter tuning  
- Add a Flask/FastAPI web app  
- Add model explainability using SHAP or LIME  
- Deploy model with Streamlit  

---

## 🤝 **Contributions**

Pull requests are welcome!  
Suggestions for improvement are encouraged.

---

## 📬 **Contact**

If you have any questions, feel free to reach out!

**GitHub:** https://github.com/harinirraj0411


## 📁 **Folder Structure**

