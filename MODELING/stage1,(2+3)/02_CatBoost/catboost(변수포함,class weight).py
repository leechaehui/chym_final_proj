# ==========================================================
# AKI Stage Binary Classification
# Stage1 vs (Stage2 + Stage3)
#
# XGBoost + Optuna
#
# Part 1/3
# ==========================================================

import os
import json
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import optuna

from catboost import CatBoostClassifier

from sklearn.utils.class_weight import (
    compute_class_weight
)

from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,

    classification_report,
    confusion_matrix,

    roc_curve,
    roc_auc_score,

    precision_recall_curve
)

# ==========================================================
# outputs 폴더 생성
# ==========================================================

os.makedirs(
    "outputs2",
    exist_ok=True
)

# ==========================================================
# 데이터 로딩
# ==========================================================

import os

print("=" * 70)
print("DATA LOADING")
print("=" * 70)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "..",
        "..",
        "data"
    )
)

print("BASE_DIR :", BASE_DIR)
print("DATA_DIR :", DATA_DIR)

X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)

X_valid = np.load(
    os.path.join(DATA_DIR, "X_valid.npy")
)

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

y_stage_train = np.load(
    os.path.join(DATA_DIR, "y_stage_train.npy")
)

y_stage_valid = np.load(
    os.path.join(DATA_DIR, "y_stage_valid.npy")
)

y_stage_test = np.load(
    os.path.join(DATA_DIR, "y_stage_test.npy")
)

print("X_train :", X_train.shape)
print("X_valid :", X_valid.shape)
print("X_test  :", X_test.shape)

print()

# ==========================================================
# AKI 환자만 추출
# Stage0 제거
# ==========================================================

train_mask = y_stage_train > 0
valid_mask = y_stage_valid > 0
test_mask = y_stage_test > 0

X_train_bin = X_train[train_mask]
X_valid_bin = X_valid[valid_mask]
X_test_bin = X_test[test_mask]

y_train_bin = y_stage_train[train_mask]
y_valid_bin = y_stage_valid[valid_mask]
y_test_bin = y_stage_test[test_mask]

print("=" * 70)
print("AKI PATIENTS ONLY")
print("=" * 70)

print("Train :", X_train_bin.shape)
print("Valid :", X_valid_bin.shape)
print("Test  :", X_test_bin.shape)

print()

# ==========================================================
# 라벨 변환
#
# Stage1      -> 0
# Stage2+3    -> 1
# ==========================================================

y_train_bin = np.where(
    y_train_bin == 1,
    0,
    1
)

y_valid_bin = np.where(
    y_valid_bin == 1,
    0,
    1
)

y_test_bin = np.where(
    y_test_bin == 1,
    0,
    1
)

print("=" * 70)
print("LABEL MAPPING")
print("=" * 70)

print("0 -> Stage1")
print("1 -> Stage2+3")

print()

# ==========================================================
# 클래스 분포
# ==========================================================

print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

unique, counts = np.unique(
    y_train_bin,
    return_counts=True
)

for cls, cnt in zip(
    unique,
    counts
):

    label = (
        "Stage1"
        if cls == 0
        else "Stage2+3"
    )

    print(
        f"{label:<15}"
        f"{cnt:>6,}"
        f" ({cnt/len(y_train_bin):.3f})"
    )

print()

# ==========================================================
# Class Weight
# ==========================================================

classes = np.unique(
    y_train_bin
)

weights = compute_class_weight(

    class_weight="balanced",

    classes=classes,

    y=y_train_bin
)

class_weight_dict = dict(
    zip(classes, weights)
)

print("=" * 70)
print("CLASS WEIGHT")
print("=" * 70)

print(class_weight_dict)

print()

scale_pos_weight = (

    class_weight_dict[1]

    /

    class_weight_dict[0]

)

print(
    "scale_pos_weight :",
    round(scale_pos_weight, 4)
)

print()

# ==========================================================
# Optuna Objective
# ==========================================================

def objective(trial):

    model = CatBoostClassifier(

        iterations=trial.suggest_int(
            "iterations",
            300,
            1000
        ),

        depth=trial.suggest_int(
            "depth",
            4,
            10
        ),

        learning_rate=trial.suggest_float(
            "learning_rate",
            0.01,
            0.2,
            log=True
        ),

        l2_leaf_reg=trial.suggest_float(
            "l2_leaf_reg",
            1,
            10
        ),

        random_strength=trial.suggest_float(
            "random_strength",
            0,
            5
        ),

        bagging_temperature=trial.suggest_float(
            "bagging_temperature",
            0,
            5
        ),

        loss_function="Logloss",

        eval_metric="F1",

        class_weights=[

            class_weight_dict[0],

            class_weight_dict[1]

        ],

        random_seed=42,

        verbose=False
    )

    model.fit(
        X_train_bin,
        y_train_bin
    )

    pred = model.predict(
        X_valid_bin
    )

    pred = pred.astype(int)

    return f1_score(
        y_valid_bin,
        pred
    )

# ==========================================================
# Optuna
# ==========================================================

print("=" * 70)
print("OPTUNA START")
print("=" * 70)

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=50
)

print()

print("=" * 70)
print("BEST SCORE")
print("=" * 70)

print(study.best_value)

print()

print("=" * 70)
print("BEST PARAMS")
print("=" * 70)

print(
    study.best_params
)

print()

with open(
    "outputs2/best_params.json",
    "w"
) as f:

    json.dump(
        study.best_params,
        f,
        indent=4
    )

# ==========================================================
# Part 2/3
#
# 최종 모델 학습
# Validation / Test 평가
# CSV 저장
# ==========================================================

# ==========================================================
# 최종 모델 학습
# ==========================================================

print("=" * 70)
print("FINAL MODEL TRAINING")
print("=" * 70)

best_params = study.best_params

cat = CatBoostClassifier(

    loss_function="Logloss",

    eval_metric="F1",

    class_weights=[

        class_weight_dict[0],

        class_weight_dict[1]

    ],

    random_seed=42,

    verbose=False,

    **best_params
)

cat.fit(
    X_train_bin,
    y_train_bin
)

print("Model Training Complete")
print()

# ==========================================================
# 모델 저장
# ==========================================================

with open(
    "outputs2/catboost_stage1_vs_stage23.pkl",
    "wb"
) as f:

    pickle.dump(
        cat,
        f
    )

print(
    "Model Saved"
)

print()

# ==========================================================
# Validation Prediction
# ==========================================================

valid_pred = cat.predict(
    X_valid_bin
)

valid_pred = valid_pred.astype(int)

valid_proba = cat.predict_proba(
    X_valid_bin
)[:,1]

# ==========================================================
# Validation Metrics
# ==========================================================

valid_acc = accuracy_score(
    y_valid_bin,
    valid_pred
)

valid_prec = precision_score(
    y_valid_bin,
    valid_pred
)

valid_rec = recall_score(
    y_valid_bin,
    valid_pred
)

valid_f1 = f1_score(
    y_valid_bin,
    valid_pred
)

print("=" * 70)
print("VALIDATION RESULT")
print("=" * 70)

print(
    f"Accuracy  : {valid_acc:.4f}"
)

print(
    f"Precision : {valid_prec:.4f}"
)

print(
    f"Recall    : {valid_rec:.4f}"
)

print(
    f"F1 Score  : {valid_f1:.4f}"
)

print()

print(
    classification_report(

        y_valid_bin,

        valid_pred,

        target_names=[
            "Stage1",
            "Stage2+3"
        ]
    )
)

# ==========================================================
# Validation Metrics 저장
# ==========================================================

valid_metrics = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1"

    ],

    "Score":[

        valid_acc,

        valid_prec,

        valid_rec,

        valid_f1
    ]
})

valid_metrics.to_csv(

    "outputs2/validation_metrics.csv",

    index=False
)

# ==========================================================
# Validation Classification Report 저장
# ==========================================================

valid_report = pd.DataFrame(

    classification_report(

        y_valid_bin,

        valid_pred,

        target_names=[
            "Stage1",
            "Stage2+3"
        ],

        output_dict=True

    )

).transpose()

valid_report.to_csv(

    "outputs2/classification_report_valid.csv"
)

# ==========================================================
# Test Prediction
# ==========================================================

test_pred = cat.predict(
    X_test_bin
)

test_pred = test_pred.astype(int)

test_proba = cat.predict_proba(
    X_test_bin
)[:,1]

# ==========================================================
# Test Metrics
# ==========================================================

test_acc = accuracy_score(
    y_test_bin,
    test_pred
)

test_prec = precision_score(
    y_test_bin,
    test_pred
)

test_rec = recall_score(
    y_test_bin,
    test_pred
)

test_f1 = f1_score(
    y_test_bin,
    test_pred
)

print()

print("=" * 70)
print("TEST RESULT")
print("=" * 70)

print(
    f"Accuracy  : {test_acc:.4f}"
)

print(
    f"Precision : {test_prec:.4f}"
)

print(
    f"Recall    : {test_rec:.4f}"
)

print(
    f"F1 Score  : {test_f1:.4f}"
)

print()

print(

    classification_report(

        y_test_bin,

        test_pred,

        target_names=[
            "Stage1",
            "Stage2+3"
        ]
    )
)

# ==========================================================
# Test Metrics 저장
# ==========================================================

test_metrics = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1"

    ],

    "Score":[

        test_acc,

        test_prec,

        test_rec,

        test_f1
    ]
})

test_metrics.to_csv(

    "outputs2/test_metrics.csv",

    index=False
)

# ==========================================================
# Test Classification Report 저장
# ==========================================================

test_report = pd.DataFrame(

    classification_report(

        y_test_bin,

        test_pred,

        target_names=[
            "Stage1",
            "Stage2+3"
        ],

        output_dict=True

    )

).transpose()

test_report.to_csv(

    "outputs2/classification_report_test.csv"
)

print()
print("=" * 70)
print("CSV SAVE COMPLETE")
print("=" * 70)
print()

# ==========================================================
# Part 3/3
#
# Visualization
# Feature Importance
# ==========================================================

print("=" * 70)
print("VISUALIZATION START")
print("=" * 70)

# ==========================================================
# Validation Confusion Matrix
# ==========================================================

cm_valid = confusion_matrix(
    y_valid_bin,
    valid_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(

    cm_valid,

    annot=True,
    fmt="d",

    cmap="Blues",

    xticklabels=[
        "Stage1",
        "Stage2+3"
    ],

    yticklabels=[
        "Stage1",
        "Stage2+3"
    ]
)

plt.title(
    "Validation Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()

plt.savefig(

    "outputs2/confusion_matrix_valid.png",

    dpi=300
)

plt.close()

# ==========================================================
# Test Confusion Matrix
# ==========================================================

cm_test = confusion_matrix(
    y_test_bin,
    test_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(

    cm_test,

    annot=True,
    fmt="d",

    cmap="Blues",

    xticklabels=[
        "Stage1",
        "Stage2+3"
    ],

    yticklabels=[
        "Stage1",
        "Stage2+3"
    ]
)

plt.title(
    "Test Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()

plt.savefig(

    "outputs2/confusion_matrix_test.png",

    dpi=300
)

plt.close()

# ==========================================================
# Validation ROC Curve
# ==========================================================

fpr, tpr, _ = roc_curve(
    y_valid_bin,
    valid_proba
)

valid_auc = roc_auc_score(
    y_valid_bin,
    valid_proba
)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {valid_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "Validation ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(

    "outputs2/roc_curve_valid.png",

    dpi=300
)

plt.close()

# ==========================================================
# Test ROC Curve
# ==========================================================

fpr, tpr, _ = roc_curve(
    y_test_bin,
    test_proba
)

test_auc = roc_auc_score(
    y_test_bin,
    test_proba
)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {test_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "Test ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(

    "outputs2/roc_curve_test.png",

    dpi=300
)

plt.close()

# ==========================================================
# Validation Precision Recall Curve
# ==========================================================

precision, recall, _ = precision_recall_curve(
    y_valid_bin,
    valid_proba
)

plt.figure(figsize=(6, 5))

plt.plot(
    recall,
    precision,
    linewidth=2
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Validation Precision Recall Curve"
)

plt.tight_layout()

plt.savefig(

    "outputs2/pr_curve_valid.png",

    dpi=300
)

plt.close()

# ==========================================================
# Test Precision Recall Curve
# ==========================================================

precision, recall, _ = precision_recall_curve(
    y_test_bin,
    test_proba
)

plt.figure(figsize=(6, 5))

plt.plot(
    recall,
    precision,
    linewidth=2
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Test Precision Recall Curve"
)

plt.tight_layout()

plt.savefig(

    "outputs2/pr_curve_test.png",

    dpi=300
)

plt.close()

# ==========================================================
# Feature Names
# ==========================================================

FEATURE_NAMES = [

    'map_mean',
    'map_min',
    'map_below65_hours',

    'sbp_min',
    'sbp_mean',
    'shock_index_mean',

    'hr_max',
    'hr_mean',

    'rr_max',
    'rr_mean',

    'temp_max',
    'temp_mean',

    'urine_output_sum',
    'urine_output_6h',
    'oliguria_flag',

    'creatinine_min',
    'creatinine_max',
    'creatinine_delta',

    'bun_max',
    'bun_cr_ratio',

    'lactate_max',
    'lactate_mean',

    'vasopressor_flag',
    'vasopressor_hours',
    'norepi_dose_max',

    'potassium_max',
    'potassium_mean',

    'bicarbonate_min',
    'bicarbonate_mean',

    'sodium_min',
    'sodium_max',

    'hemoglobin_min',
    'hemoglobin_mean',

    'spo2_min',
    'spo2_mean'
]

# ==========================================================
# Feature Importance
# ==========================================================

importance = pd.DataFrame({

    "Feature": FEATURE_NAMES,

    "Importance":
        cat.get_feature_importance()
})

importance = importance.sort_values(

    "Importance",

    ascending=False
)

importance.to_csv(

    "outputs2/feature_importance.csv",

    index=False
)

# ==========================================================
# Top15 Importance Plot
# ==========================================================

top15 = importance.head(15)

plt.figure(figsize=(8, 6))

plt.barh(

    top15["Feature"],

    top15["Importance"]
)

plt.gca().invert_yaxis()

plt.title(
    "Feature Importance Top15"
)

plt.xlabel(
    "Importance"
)

plt.tight_layout()

plt.savefig(

    "outputs2/feature_importance_top15.png",

    dpi=300
)

plt.close()

# ==========================================================
# Summary
# ==========================================================

print()
print("=" * 70)
print("ALL FILES SAVED")
print("=" * 70)

print("best_params.json")
print("validation_metrics.csv")
print("test_metrics.csv")

print("classification_report_valid.csv")
print("classification_report_test.csv")

print("confusion_matrix_valid.png")
print("confusion_matrix_test.png")

print("roc_curve_valid.png")
print("roc_curve_test.png")

print("pr_curve_valid.png")
print("pr_curve_test.png")

print("feature_importance.csv")
print("feature_importance_top15.png")

print("catboost_stage1_vs_stage23.pkl")

print()
print("=" * 70)
print("FINISHED")
print("=" * 70)