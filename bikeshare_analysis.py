# ============================================================
# Washington DC Bike Share: Count Regression
# STAT 4000 Portfolio Project 3 — Python Implementation
# Author: Jrudani21
# ============================================================
# Dataset: UCI Bike Sharing Dataset
# Download: https://archive.uci.edu/dataset/275/bike+sharing+dataset
# Place day.csv and hour.csv in data/ folder
# ============================================================

# %% [markdown]
# ## Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## 1. Load & Prepare Data

# %%
day  = pd.read_csv("data/day.csv")
hour = pd.read_csv("data/hour.csv")

# Decode factors
season_map  = {1:"Spring", 2:"Summer", 3:"Fall", 4:"Winter"}
weather_map = {1:"Clear", 2:"Mist", 3:"Light Rain", 4:"Heavy Rain"}
yr_map      = {0:"2011", 1:"2012"}

day["season_lbl"]  = day["season"].map(season_map)
day["weather_lbl"] = day["weathersit"].map(weather_map)
day["yr_lbl"]      = day["yr"].map(yr_map)
day["workingday_lbl"] = day["workingday"].map(
    {0:"Weekend/Holiday", 1:"Working Day"})

print(f"Daily records:  {len(day)}")
print(f"Hourly records: {len(hour)}")
print(f"\nDaily cnt  —  Mean: {day['cnt'].mean():.1f}  |  "
      f"Var: {day['cnt'].var():.1f}  |  "
      f"Ratio: {day['cnt'].var()/day['cnt'].mean():.1f}")

# %% [markdown]
# ## 2. EDA

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram of daily rentals
axes[0,0].hist(day["cnt"], bins=30, color="steelblue",
               edgecolor="white", alpha=0.85)
axes[0,0].set_title("Distribution of Daily Rentals")
axes[0,0].set_xlabel("Daily Rentals")
axes[0,0].set_ylabel("Count")

# Boxplot by season
season_order = ["Spring","Summer","Fall","Winter"]
sns.boxplot(data=day, x="season_lbl", y="cnt",
            order=season_order, palette="Set2", ax=axes[0,1])
axes[0,1].set_title("Daily Rentals by Season")
axes[0,1].set_xlabel("Season")
axes[0,1].set_ylabel("Daily Rentals")

# Hourly pattern (working vs non-working)
hour["workingday_lbl"] = hour["workingday"].map(
    {0:"Weekend/Holiday", 1:"Working Day"})
hr_avg = hour.groupby(["hr","workingday_lbl"])["cnt"].mean().reset_index()
for label, grp in hr_avg.groupby("workingday_lbl"):
    axes[1,0].plot(grp["hr"], grp["cnt"], marker="o",
                   markersize=4, lw=2, label=label)
axes[1,0].set_title("Avg Hourly Rentals: Working vs Weekend")
axes[1,0].set_xlabel("Hour of Day")
axes[1,0].set_ylabel("Mean Rentals")
axes[1,0].legend()
axes[1,0].set_xticks(range(0,24))

# Temp vs cnt scatter
for season in season_order:
    subset = day[day["season_lbl"] == season]
    axes[1,1].scatter(subset["temp"]*41, subset["cnt"],
                      alpha=0.4, label=season, s=20)
axes[1,1].set_title("Daily Rentals vs Temperature")
axes[1,1].set_xlabel("Temperature (°C)")
axes[1,1].set_ylabel("Daily Rentals")
axes[1,1].legend(title="Season", fontsize=8)

plt.tight_layout()
plt.savefig("figures/py_01_eda.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Mean–Variance Check (Poisson Assumption)

# %%
mean_cnt = day["cnt"].mean()
var_cnt  = day["cnt"].var()
ratio    = var_cnt / mean_cnt

print("=" * 45)
print(f"{'Mean of cnt:':<30} {mean_cnt:.2f}")
print(f"{'Variance of cnt:':<30} {var_cnt:.2f}")
print(f"{'Variance / Mean ratio:':<30} {ratio:.2f}")
print("=" * 45)
print("Ratio >> 1 → overdispersion present")
print("→ Negative Binomial preferred over Poisson")

# %% [markdown]
# ## 4. Poisson GLM

# %%
poisson_model = smf.glm(
    "cnt ~ temp + hum + windspeed + C(workingday) + "
    "C(season) + C(weathersit) + C(yr)",
    data   = day,
    family = sm.families.Poisson()
).fit()

print(poisson_model.summary())

# Pearson chi² / df  (overdispersion check)
pearson_chi2 = poisson_model.pearson_chi2
df_resid     = poisson_model.df_resid
print(f"\nPearson χ²/df = {pearson_chi2/df_resid:.2f}  (>1 = overdispersed)")

# %% [markdown]
# ## 5. Negative Binomial GLM

# %%
# Fit NB to get alpha (overdispersion parameter)
nb_fit = sm.NegativeBinomial(
    day["cnt"],
    sm.add_constant(pd.get_dummies(
        day[["temp","hum","windspeed","workingday",
             "season","weathersit","yr"]],
        columns=["workingday","season","weathersit","yr"],
        drop_first=True
    ))
).fit(disp=False)

print(nb_fit.summary())

# Using formula API for cleaner output
nb_model = smf.glm(
    "cnt ~ temp + hum + windspeed + C(workingday) + "
    "C(season) + C(weathersit) + C(yr)",
    data   = day,
    family = sm.families.NegativeBinomial(alpha=nb_fit.params["alpha"])
).fit()

print(nb_model.summary())

# %% [markdown]
# ## 6. Model Comparison

# %%
print("=" * 45)
print(f"{'Model':<30} {'AIC':>10}")
print("-" * 45)
print(f"{'Poisson GLM':<30} {poisson_model.aic:>10.1f}")
print(f"{'Negative Binomial':<30} {nb_model.aic:>10.1f}")
print("=" * 45)
print("Lower AIC = better fit → Negative Binomial wins")

# %% [markdown]
# ## 7. Incidence Rate Ratios (IRRs)

# %%
irr  = np.exp(nb_model.params)
ci   = np.exp(nb_model.conf_int())
pval = nb_model.pvalues

irr_df = pd.DataFrame({
    "IRR":     irr,
    "CI_low":  ci[0],
    "CI_high": ci[1],
    "p_value": pval
}).drop("Intercept").sort_values("IRR", ascending=False)

print("Incidence Rate Ratios (Negative Binomial):")
print(irr_df.round(4))

# %%
# IRR forest plot
irr_sig = irr_df.sort_values("IRR")

fig, ax = plt.subplots(figsize=(9, max(6, len(irr_sig)*0.4)))
ax.scatter(irr_sig["IRR"], range(len(irr_sig)),
           color="steelblue", s=60, zorder=3)
for i, (_, row) in enumerate(irr_sig.iterrows()):
    ax.plot([row["CI_low"], row["CI_high"]], [i, i],
            color="steelblue", lw=1.5)
ax.axvline(1, color="red", linestyle="--", lw=1.5)
ax.set_yticks(range(len(irr_sig)))
ax.set_yticklabels(irr_sig.index, fontsize=8)
ax.set_xlabel("Incidence Rate Ratio (95% CI)")
ax.set_title("IRRs — Negative Binomial Model\n"
             "IRR > 1 = more rentals | IRR < 1 = fewer rentals")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/py_02_irr_plot.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8. Predicted vs Actual

# %%
day["predicted"] = nb_model.fittedvalues

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(day["predicted"], day["cnt"],
           alpha=0.4, color="steelblue", s=20)
lims = [min(day["predicted"].min(), day["cnt"].min()),
        max(day["predicted"].max(), day["cnt"].max())]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
ax.set_title("Predicted vs Actual Daily Rentals\n(Negative Binomial)")
ax.set_xlabel("Predicted Rentals")
ax.set_ylabel("Actual Rentals")
ax.legend()
plt.tight_layout()
plt.savefig("figures/py_03_pred_vs_actual.png", dpi=150)
plt.show()

# %% [markdown]
# ## 9. Residual Diagnostics

# %%
pearson_resids = nb_model.resid_pearson

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(nb_model.fittedvalues, pearson_resids,
                alpha=0.4, color="steelblue", s=20)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title("Residual Plot (Pearson)")
axes[0].set_xlabel("Fitted Values")
axes[0].set_ylabel("Pearson Residuals")

sm.qqplot(pearson_resids, line="s", ax=axes[1], alpha=0.4)
axes[1].set_title("Normal Q-Q (Pearson Residuals)")

plt.tight_layout()
plt.savefig("figures/py_04_diagnostics.png", dpi=150)
plt.show()

# %% [markdown]
# ## 10. Results Summary

# %%
print("=" * 55)
print(f"{'Metric':<35} Value")
print("-" * 55)
print(f"{'Variance/Mean ratio':<35} {ratio:.1f}  → overdispersed")
print(f"{'Poisson AIC':<35} {poisson_model.aic:.1f}")
print(f"{'Negative Binomial AIC':<35} {nb_model.aic:.1f}  (better)")
print("=" * 55)

print("\nKey IRR Insights:")
print("• 2012 vs 2011: ~54% more rentals (system growth)")
print("• Fall vs Spring: most rentals in Fall")
print("• Higher temperature → more rentals")
print("• Light rain → ~30% fewer rentals vs clear sky")
print("• Higher humidity → fewer rentals")
print("• Working days vs weekends: different commute patterns")
