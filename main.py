import pandas as pd

df = pd.read_csv("final_features_48h.csv")

# stay_id별 행 개수 확인
stay_counts = df.groupby("stay_id").size()

print(stay_counts.describe())

# 1개 초과인 stay_id 개수
multi_stay = stay_counts[stay_counts > 1]

print("여러 행을 가진 stay_id 수:", len(multi_stay))
print(multi_stay.head())

print(df.columns)
