import pandas as pd
from statsmodels.stats.anova import AnovaRM

data = pd.read_csv("combined.csv")

# keep only main trials
data = data[data["block"] == "main"]

# -------------------------
# RT = correct trials only
# -------------------------
rt_data = data[data["accuracy"] == 1].copy()
rt_data["rt"] = pd.to_numeric(rt_data["rt"], errors="coerce")
rt_data = rt_data.dropna(subset=["rt"])

rt_results = AnovaRM(
    data=rt_data,
    depvar="rt",
    subject="participant",
    within=["type"],
    aggregate_func="mean"
).fit()

print("\nRT 1-way:")
print(rt_results)

# -------------------------
# Accuracy = all trials
# -------------------------
acc_results = AnovaRM(
    data=data,
    depvar="accuracy",
    subject="participant",
    within=["type"],
    aggregate_func="mean"
).fit()

print("\nAccuracy 1-way:")
print(acc_results)