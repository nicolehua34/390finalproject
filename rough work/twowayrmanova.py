import pandas as pd
from statsmodels.stats.anova import AnovaRM

data = pd.read_csv("combined.csv")
data = data[data["block"] == "main"]

# -------------------------
# RT (correct trials only)
# -------------------------
rt = data[data["accuracy"] == 1].copy()

rt_agg = rt.groupby(["participant", "type", "direction"], as_index=False)["rt"].mean()

# keep only participants with all 4 conditions
rt_agg = rt_agg.groupby("participant").filter(lambda x: len(x) == 4)

rt_results = AnovaRM(
    data=rt_agg,
    depvar="rt",
    subject="participant",
    within=["type", "direction"]
).fit()

print("\nRT 2x2:")
print(rt_results)

# -------------------------
# Accuracy (all trials)
# -------------------------
acc_agg = data.groupby(["participant", "type", "direction"], as_index=False)["accuracy"].mean()

acc_agg = acc_agg.groupby("participant").filter(lambda x: len(x) == 4)

acc_results = AnovaRM(
    data=acc_agg,
    depvar="accuracy",
    subject="participant",
    within=["type", "direction"]
).fit()

print("\nAccuracy 2x2:")
print(acc_results)