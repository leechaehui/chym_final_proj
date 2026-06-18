# ============================================================
# AKI Statistical Significance Analysis
# Final Version
# ============================================================

# 포함 내용
# 1. EDA
# 2. Missing analysis
# 3. Outlier analysis
# 4. Normality test
# 5. Mann-Whitney U test
# 6. Chi-square test
# 7. FDR correction
# 8. Effect size (Cohen's d)
# 9. Baseline table
# 10. Important feature extraction
# 11. CSV save
# ============================================================


# ============================================================
# 1. Library Import
# ============================================================

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import (
    shapiro,
    mannwhitneyu,
    chi2_contingency
)

from statsmodels.stats.multitest import multipletests

import os


# ============================================================
# 2. CSV Load
# ============================================================

file_path = "final_features_48h.csv"

df = pd.read_csv(file_path)


# ============================================================
# 3. Basic EDA
# ============================================================

print("\n===== DATA SHAPE =====")
print(df.shape)

print("\n===== HEAD =====")
print(df.head())

print("\n===== INFO =====")
print(df.info())

print("\n===== DESCRIBE =====")
print(df.describe())


# ============================================================
# 4. AKI Label Distribution
# ============================================================

print("\n===== AKI LABEL DISTRIBUTION =====")

print(df['aki_label'].value_counts())

print("\n===== AKI LABEL RATIO =====")

print(df['aki_label'].value_counts(normalize=True))


# ============================================================
# 5. Missing Value Analysis
# ============================================================

print("\n===== MISSING VALUES =====")

missing = df.isnull().sum().sort_values(
    ascending=False
)

print(missing)

print("\n===== MISSING RATIO (%) =====")

missing_ratio = (
    df.isnull().mean() * 100
).sort_values(ascending=False)

print(missing_ratio)


# ============================================================
# 6. AKI / Non-AKI Split
# ============================================================

aki_df = df[df['aki_label'] == 1]

non_aki_df = df[df['aki_label'] == 0]

print("\n===== GROUP SHAPE =====")

print("AKI:", aki_df.shape)

print("Non-AKI:", non_aki_df.shape)


# ============================================================
# 7. Column List
# ============================================================

print("\n===== COLUMN LIST =====")

print(df.columns.tolist())


# ============================================================
# 8. Exclude Columns
# ============================================================

exclude_cols = [

    'stay_id',

    'subject_id',

    'hadm_id',

    'aki_label',

    'aki_onset_time',

    'prediction_cutoff',

    'index_time'
]


# ============================================================
# 9. Categorical Columns
# ============================================================

categorical_cols = [

    'gender',

    'aki_stage',

    'oliguria_flag',

    'vasopressor_flag'
]


# ============================================================
# 10. Continuous Columns
# ============================================================

continuous_cols = [

    'age',

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

    'urine_ml_kg_hr',

    'creatinine_min',

    'creatinine_max',

    'creatinine_delta',

    'bun_max',

    'bun_cr_ratio',

    'lactate_max',

    'lactate_mean',

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


# ============================================================
# 11. String Columns Check
# ============================================================

print("\n===== STRING COLUMNS =====")

string_cols = df.select_dtypes(
    include=['object', 'string']
).columns.tolist()

print(string_cols)


# ============================================================
# 12. Binary Column Detection
# ============================================================

print("\n===== BINARY COLUMNS =====")

binary_cols = []

for col in df.columns:

    unique_values = df[col].dropna().unique()

    if len(unique_values) <= 2:

        binary_cols.append(col)

print(binary_cols)


# ============================================================
# 13. Outlier Analysis
# IQR Method
# ============================================================

outlier_results = []

for col in continuous_cols:

    values = df[col].dropna()

    # 데이터 부족 시 skip
    if len(values) < 10:
        continue

    try:

        # Q1 / Q3
        Q1 = values.quantile(0.25)

        Q3 = values.quantile(0.75)

        # IQR 계산
        IQR = Q3 - Q1

        # 경계값
        lower = Q1 - 1.5 * IQR

        upper = Q3 + 1.5 * IQR

        # 이상치 추출
        outliers = values[
            (values < lower) |
            (values > upper)
        ]

        # 이상치 개수
        outlier_count = len(outliers)

        # 전체 개수
        total_count = len(values)

        # 이상치 비율
        outlier_ratio = (
            outlier_count / total_count
        ) * 100

        outlier_results.append({

            'variable': col,

            'Q1': Q1,

            'Q3': Q3,

            'IQR': IQR,

            'lower_bound': lower,

            'upper_bound': upper,

            'outlier_count': outlier_count,

            'total_count': total_count,

            'outlier_ratio_percent': round(
                outlier_ratio,
                2
            )
        })

    except Exception as e:

        print(f"ERROR in {col}: {e}")


# ============================================================
# 14. Outlier DataFrame
# ============================================================

outlier_df = pd.DataFrame(
    outlier_results
)

outlier_df = outlier_df.sort_values(
    by='outlier_ratio_percent',
    ascending=False
)

print("\n===== OUTLIER RESULTS =====")

print(outlier_df)


# ============================================================
# 15. Boxplot Save
# ============================================================

os.makedirs(
    'boxplots',
    exist_ok=True
)

for col in continuous_cols:

    try:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            x=df[col].dropna()
        )

        plt.title(f"{col} Boxplot")

        plt.tight_layout()

        plt.savefig(
            f'boxplots/{col}_boxplot.png'
        )

        plt.close()

    except Exception as e:

        print(f"ERROR in plot {col}: {e}")

print("\n===== BOXPLOT SAVE COMPLETE =====")


# ============================================================
# 16. Normality Test
# ============================================================

sample_data = df['map_mean'].dropna().sample(
    5000,
    random_state=42
)

stat, p = shapiro(sample_data)

print("\n===== SHAPIRO TEST =====")

print("Statistic:", stat)

print("P-value:", p)

# p <= 0.05
# -> 비정규분포
# -> Mann-Whitney 사용


# ============================================================
# 17. Continuous Variable Test
# Mann-Whitney U
# ============================================================

results = []

for col in continuous_cols:

    aki_values = aki_df[col].dropna()

    non_aki_values = non_aki_df[col].dropna()

    # 데이터 부족 시 skip
    if len(aki_values) < 3 or len(non_aki_values) < 3:
        continue

    try:

        stat, p = mannwhitneyu(

            aki_values,

            non_aki_values,

            alternative='two-sided'
        )

        results.append({

            'variable': col,

            'test': 'Mann-Whitney U',

            'p_value': p,

            'AKI_median': aki_values.median(),

            'NonAKI_median': non_aki_values.median()
        })

    except Exception as e:

        print(f"ERROR in {col}: {e}")


# ============================================================
# 18. Categorical Variable Test
# Chi-square
# ============================================================

categorical_results = []

for col in categorical_cols:

    if col == 'aki_label':
        continue

    contingency = pd.crosstab(
        df[col],
        df['aki_label']
    )

    try:

        chi2, p, dof, expected = chi2_contingency(
            contingency
        )

        categorical_results.append({

            'variable': col,

            'test': 'Chi-square',

            'p_value': p
        })

    except Exception as e:

        print(f"ERROR in {col}: {e}")


# ============================================================
# 19. Final Result Merge
# ============================================================

continuous_results_df = pd.DataFrame(results)

categorical_results_df = pd.DataFrame(
    categorical_results
)

final_results = pd.concat(
    [
        continuous_results_df,
        categorical_results_df
    ],
    ignore_index=True
)

final_results = final_results.sort_values(
    'p_value'
)

print("\n===== FINAL RESULTS =====")

print(final_results)


# ============================================================
# 20. FDR Correction
# ============================================================

pvals = final_results['p_value']

reject, corrected_pvals, _, _ = multipletests(

    pvals,

    method='fdr_bh'
)

final_results['fdr_p'] = corrected_pvals

final_results['significant'] = reject

print("\n===== AFTER FDR =====")

print(final_results)


# ============================================================
# 21. Cohen's d Function
# ============================================================

def cohens_d(x, y):

    nx = len(x)

    ny = len(y)

    pooled_std = np.sqrt(

        (
            ((nx - 1) * np.var(x, ddof=1)) +

            ((ny - 1) * np.var(y, ddof=1))
        ) / (nx + ny - 2)
    )

    # 0 division 방지
    if pooled_std == 0:

        return np.nan

    return (
        np.mean(x) - np.mean(y)
    ) / pooled_std


# ============================================================
# 22. Effect Size Analysis
# ============================================================

effect_size_results = []

for col in continuous_cols:

    aki_values = aki_df[col].dropna()

    non_aki_values = non_aki_df[col].dropna()

    if len(aki_values) < 3 or len(non_aki_values) < 3:
        continue

    try:

        d = cohens_d(
            aki_values,
            non_aki_values
        )

        abs_d = abs(d)

        # effect size category
        if abs_d < 0.2:

            effect_category = "Very Small"

        elif abs_d < 0.5:

            effect_category = "Small"

        elif abs_d < 0.8:

            effect_category = "Medium"

        else:

            effect_category = "Large"

        effect_size_results.append({

            'variable': col,

            'cohens_d': d,

            'abs_cohens_d': abs_d,

            'effect_size_category': effect_category
        })

    except Exception as e:

        print(f"ERROR in effect size {col}: {e}")


# ============================================================
# 23. Effect Size DataFrame
# ============================================================

effect_size_df = pd.DataFrame(
    effect_size_results
)

effect_size_df = effect_size_df.sort_values(
    by='abs_cohens_d',
    ascending=False
)

print("\n===== EFFECT SIZE RESULTS =====")

print(effect_size_df)


# ============================================================
# 24. Merge Effect Size
# ============================================================

final_results = pd.merge(

    final_results,

    effect_size_df,

    on='variable',

    how='left'
)


# ============================================================
# 25. Baseline Table
# ============================================================

baseline_rows = []

for col in continuous_cols:

    aki_vals = aki_df[col].dropna()

    non_vals = non_aki_df[col].dropna()

    if len(aki_vals) < 3 or len(non_vals) < 3:
        continue

    try:

        _, p = mannwhitneyu(
            aki_vals,
            non_vals
        )

        baseline_rows.append({

            'Variable': col,

            'AKI':
                f"{aki_vals.median():.2f} "
                f"({aki_vals.quantile(0.25):.2f}-"
                f"{aki_vals.quantile(0.75):.2f})",

            'Non_AKI':
                f"{non_vals.median():.2f} "
                f"({non_vals.quantile(0.25):.2f}-"
                f"{non_vals.quantile(0.75):.2f})",

            'p_value': p
        })

    except Exception as e:

        print(f"ERROR in baseline {col}: {e}")


baseline_table = pd.DataFrame(
    baseline_rows
)

print("\n===== BASELINE TABLE =====")

print(baseline_table)


# ============================================================
# 26. Important Feature Extraction
# 조건:
# 1. FDR significant
# 2. effect size >= 0.2
# ============================================================

important_features = final_results[

    (final_results['fdr_p'] < 0.05) &

    (final_results['abs_cohens_d'] >= 0.2)

].copy()

important_features = important_features.sort_values(
    by='abs_cohens_d',
    ascending=False
)

print("\n===== IMPORTANT FEATURES =====")

print(

    important_features[
        [
            'variable',
            'p_value',
            'fdr_p',
            'cohens_d',
            'effect_size_category'
        ]
    ]
)


# ============================================================
# 27. Significant Feature
# ============================================================

significant_results = final_results[
    final_results['fdr_p'] < 0.05
].copy()

significant_results = significant_results.sort_values(
    by='p_value',
    ascending=True
)

significant_results.reset_index(
    drop=True,
    inplace=True
)

print("\n===== SIGNIFICANT FEATURES =====")

print(significant_results)


# ============================================================
# 28. CSV Save
# ============================================================

final_results.to_csv(
    'statistical_test_results.csv',
    index=False
)

baseline_table.to_csv(
    'baseline_table.csv',
    index=False
)

significant_results.to_csv(
    'significant_features.csv',
    index=False
)

effect_size_df.to_csv(
    'effect_size_results.csv',
    index=False
)

important_features.to_csv(
    'important_features.csv',
    index=False
)

outlier_df.to_csv(
    'outlier_analysis.csv',
    index=False
)

print("\n===== SAVE COMPLETE =====")

print("1. statistical_test_results.csv")

print("2. baseline_table.csv")

print("3. significant_features.csv")

print("4. effect_size_results.csv")

print("5. important_features.csv")

print("6. outlier_analysis.csv")