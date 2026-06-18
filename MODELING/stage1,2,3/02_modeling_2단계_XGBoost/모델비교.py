import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# XGBoost 결과
# ==========================================================

A_macro_f1 = 0.301   # 변수포함 + Class Weight
B_macro_f1 = 0.295   # 변수제거 + Class Weight
C_macro_f1 = 0.294   # 변수제거 + SMOTE
D_macro_f1 = 0.305   # 변수포함 + SMOTE

# ==========================================================
# 1. 전체 모델 비교
# ==========================================================

compare = pd.DataFrame({

    "Model":[
        "A\nInclude\nWeight",
        "B\nRemove\nWeight",
        "C\nRemove\nSMOTE",
        "D\nInclude\nSMOTE"
    ],

    "Macro_F1":[
        A_macro_f1,
        B_macro_f1,
        C_macro_f1,
        D_macro_f1
    ]
})

plt.figure(figsize=(10,6))

bars = plt.bar(
    compare["Model"],
    compare["Macro_F1"]
)

plt.ylabel("Macro F1")
plt.title("XGBoost Model Comparison")
plt.ylim(0.25,0.35)

for bar in bars:

    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.001,
        f"{bar.get_height():.3f}",
        ha='center'
    )

plt.tight_layout()
plt.savefig(
    "xgb_model_comparison.png",
    dpi=300
)
plt.show()

# ==========================================================
# 2. KDIGO 변수 효과
# ==========================================================

kdigo = pd.DataFrame({

    "Setting":[
        "Include",
        "Remove"
    ],

    "Macro_F1":[
        (A_macro_f1 + D_macro_f1)/2,
        (B_macro_f1 + C_macro_f1)/2
    ]
})

plt.figure(figsize=(7,5))

bars = plt.bar(
    kdigo["Setting"],
    kdigo["Macro_F1"]
)

plt.ylabel("Macro F1")
plt.title("Effect of KDIGO Variables")

for bar in bars:

    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.001,
        f"{bar.get_height():.3f}",
        ha='center'
    )

plt.tight_layout()
plt.savefig(
    "kdigo_effect.png",
    dpi=300
)
plt.show()

# ==========================================================
# 3. SMOTE 효과
# ==========================================================

smote = pd.DataFrame({

    "Setting":[
        "Class Weight",
        "SMOTE"
    ],

    "Macro_F1":[
        (A_macro_f1 + B_macro_f1)/2,
        (C_macro_f1 + D_macro_f1)/2
    ]
})

plt.figure(figsize=(7,5))

bars = plt.bar(
    smote["Setting"],
    smote["Macro_F1"]
)

plt.ylabel("Macro F1")
plt.title("Effect of SMOTE")

for bar in bars:

    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.001,
        f"{bar.get_height():.3f}",
        ha='center'
    )

plt.tight_layout()
plt.savefig(
    "smote_effect.png",
    dpi=300
)
plt.show()

print("저장 완료")
print("1. xgb_model_comparison.png")
print("2. kdigo_effect.png")
print("3. smote_effect.png")