import pandas as pd 
from statsmodels.stats.anova import AnovaRM

data = pd.read_csv("combined.csv")

#descriptive statistics -- mean rt and mean accuracy 
print ("Mean RT:")
print(data.groupby("type")["rt"].mean())

print("\nAccuracy:")
print(data.groupby("type")["accuracy"].mean())


#one-way anova 

# keep only main trials
data = data[data["block"] == "main"]

# -------------------------
# RT = correct trials only
# -------------------------
rt_data = data[data["accuracy"] == 1].copy()
rt_data["rt"] = pd.to_numeric(rt_data["rt"], errors="coerce")
rt_data = rt_data.dropna(subset=["rt"])

rt_results1way = AnovaRM(
    data=rt_data,
    depvar="rt",
    subject="participant",
    within=["type"],
    aggregate_func="mean"
).fit()

print("\nRT 1-way:")
print(rt_results1way)

#in-line stat
rt1 = rt_results1way.anova_table.iloc[0]
rt1_p = rt1["Pr > F"]
rt1_p_txt = "< .001" if rt1_p < .001 else f"= {rt1_p:.3f}"

rt1_stat = (
    f"F({int(rt1['Num DF'])}, {int(rt1['Den DF'])}) = "
    f"{rt1['F Value']:.2f}, p {rt1_p_txt}"
)


# -------------------------
# Accuracy = all trials
# -------------------------
acc_results1way = AnovaRM(
    data=data,
    depvar="accuracy",
    subject="participant",
    within=["type"],
    aggregate_func="mean"
).fit()

print("\nAccuracy 1-way:")
print(acc_results1way)

#inline stat
acc1 = acc_results1way.anova_table.iloc[0]
acc1_p = acc1["Pr > F"]
acc1_p_txt = "< .001" if acc1_p < .001 else f"= {acc1_p:.3f}"

acc1_stat = (
    f"F({int(acc1['Num DF'])}, {int(acc1['Den DF'])}) = "
    f"{acc1['F Value']:.2f}, p {acc1_p_txt}"
)

#two-way anova

# -------------------------
# RT (correct trials only)
# -------------------------
rt = data[data["accuracy"] == 1].copy()

rt_agg = rt.groupby(["participant", "type", "direction"], as_index=False)["rt"].mean()

# keep only participants with all 4 conditions
rt_agg = rt_agg.groupby("participant").filter(lambda x: len(x) == 4)

rt_results2way = AnovaRM(
    data=rt_agg,
    depvar="rt",
    subject="participant",
    within=["type", "direction"]
).fit()

print("\nRT 2x2:")
print(rt_results2way)

#inline stat 
rt2_type = rt_results2way.anova_table.loc["type"]
rt2_dir = rt_results2way.anova_table.loc["direction"]
rt2_int = rt_results2way.anova_table.loc["type:direction"]

rt2_type_p = rt2_type["Pr > F"]
rt2_dir_p = rt2_dir["Pr > F"]
rt2_int_p = rt2_int["Pr > F"]

rt2_type_p_txt = "< .001" if rt2_type_p < .001 else f"= {rt2_type_p:.3f}"
rt2_dir_p_txt = "< .001" if rt2_dir_p < .001 else f"= {rt2_dir_p:.3f}"
rt2_int_p_txt = "< .001" if rt2_int_p < .001 else f"= {rt2_int_p:.3f}"

rt2_cong_stat = (
    f"F({int(rt2_type['Num DF'])}, {int(rt2_type['Den DF'])}) = "
    f"{rt2_type['F Value']:.2f}, p {rt2_type_p_txt}"
)

rt2_dir_stat = (
    f"F({int(rt2_dir['Num DF'])}, {int(rt2_dir['Den DF'])}) = "
    f"{rt2_dir['F Value']:.2f}, p {rt2_dir_p_txt}"
)

rt2_int_stat = (
    f"F({int(rt2_int['Num DF'])}, {int(rt2_int['Den DF'])}) = "
    f"{rt2_int['F Value']:.2f}, p {rt2_int_p_txt}"
)

# -------------------------
# Accuracy (all trials)
# -------------------------
acc_agg = data.groupby(["participant", "type", "direction"], as_index=False)["accuracy"].mean()

acc_agg = acc_agg.groupby("participant").filter(lambda x: len(x) == 4)

acc_results2way = AnovaRM(
    data=acc_agg,
    depvar="accuracy",
    subject="participant",
    within=["type", "direction"]
).fit()

print("\nAccuracy 2x2:")
print(acc_results2way)

#inline stat 
acc2_type = acc_results2way.anova_table.loc["type"]
acc2_dir = acc_results2way.anova_table.loc["direction"]
acc2_int = acc_results2way.anova_table.loc["type:direction"]

acc2_type_p = acc2_type["Pr > F"]
acc2_dir_p = acc2_dir["Pr > F"]
acc2_int_p = acc2_int["Pr > F"]

acc2_type_p_txt = "< .001" if acc2_type_p < .001 else f"= {acc2_type_p:.3f}"
acc2_dir_p_txt = "< .001" if acc2_dir_p < .001 else f"= {acc2_dir_p:.3f}"
acc2_int_p_txt = "< .001" if acc2_int_p < .001 else f"= {acc2_int_p:.3f}"

acc2_cong_stat = (
    f"F({int(acc2_type['Num DF'])}, {int(acc2_type['Den DF'])}) = "
    f"{acc2_type['F Value']:.2f}, p {acc2_type_p_txt}"
)

acc2_dir_stat = (
    f"F({int(acc2_dir['Num DF'])}, {int(acc2_dir['Den DF'])}) = "
    f"{acc2_dir['F Value']:.2f}, p {acc2_dir_p_txt}"
)

acc2_int_stat = (
    f"F({int(acc2_int['Num DF'])}, {int(acc2_int['Den DF'])}) = "
    f"{acc2_int['F Value']:.2f}, p {acc2_int_p_txt}"
)


#print all stats
print(rt1_stat)

print(acc1_stat)

print(rt2_cong_stat)
print(rt2_dir_stat)
print(rt2_int_stat)

print(acc2_cong_stat)
print(acc2_dir_stat)
print(acc2_int_stat)
