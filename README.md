# 🏥 AKI Early Prediction CDSS

ICU 환자의 Acute Kidney Injury (AKI)를 48시간 전에 예측하기 위한
AI 기반 Clinical Decision Support System(CDSS) 프로젝트입니다.

---

## 📌 Project Overview

- **Dataset** : MIMIC-IV
- **Prediction Target**
  - Stage 1 : AKI 발생 여부 예측
  - Stage 2 : AKI 중증도(Stage 1~3) 분류
- **Prediction Horizon** : AKI 발생 48시간 전
- **Goal** : 의료진의 조기 개입 지원 및 환자 예후 개선

---

## 🛠 Data Preprocessing

- Outlier Handling
- Missing Value Imputation
- Feature Scaling
- Class Imbalance Handling
- Exploratory Data Analysis (EDA)
- Statistical Analysis

---

## 🤖 Modeling

### Stage 1 : AKI Binary Classification
- Logistic Regression (Final Model)

### Stage 2 : AKI Severity Classification
- LightGBM (Final Model)

### Additional Experiments
- XGBoost
- CatBoost
- SMOTE
- Class Weight
- Threshold Optimization
- Hyperparameter Tuning (Optuna)

---

## 📊 Key Features

- Creatinine
- BUN / Creatinine Ratio
- MAP (Mean Arterial Pressure)
- Urine Output
- Lactate
- Hemoglobin
- Vasopressor Usage

---

## 📂 Project Structure

```text
DATA_PREPROCESSING/
FINAL_MODELING/
MODELING/
boxplots/
statistical/
```



---

## 👩‍💻 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Matplotlib
- MIMIC-IV

---

## 📈 Final Models

| Stage | Model |
|---------|---------|
| AKI Prediction | Logistic Regression |
| AKI Severity Classification | LightGBM |
