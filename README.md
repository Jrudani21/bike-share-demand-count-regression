# 🚲 Washington DC Bike Share: Count Regression (Poisson → Negative Binomial)

> **How do weather and calendar drive daily bike rental demand in Washington DC?**  
> 731 days of Capital Bikeshare data modelled with Poisson and Negative Binomial GLMs  in both R and Python.

Plots (including predicted vs actual) are written to `figures/` when you run `bikeshare_analysis.py`; they are not committed.

---

## 📋 Table of Contents
- [Problem](#problem)
- [Data](#data)
- [Why Not Linear Regression?](#why-not-linear-regression)
- [Methods](#methods)
- [Key Results](#key-results)
- [How to Reproduce](#how-to-reproduce)
- [What I'd Do Next](#what-id-do-next)

---

## Problem

Daily bike rentals are **count data**  non-negative integers that can't be negative and tend to be right-skewed. Using ordinary linear regression would be the wrong tool. This project:

1. Confirms that a Poisson GLM is appropriate (or detects overdispersion requiring Negative Binomial)
2. Fits both models and compares them with AIC
3. Interprets **Incidence Rate Ratios (IRRs)** to quantify the effect of weather, season, and day type
4. Validates assumptions with residual diagnostics

---

## Data

| Property | Detail |
|---|---|
| **Source** | [UCI Machine Learning Repository  Bike Sharing Dataset](https://archive.uci.edu/dataset/275/bike+sharing+dataset) |
| **Collector** | Fanaee-T, Hadi & Gama, João (2013) |
| **Size** | `day.csv`  731 rows × 16 columns; `hour.csv`  17,379 rows × 17 columns |
| **Target** | `cnt`  total daily rentals (casual + registered) |
| **Period** | Jan 1 2011 – Dec 31 2012, Washington DC |
| **License** | CC BY 4.0 |

### Key Variables

| Variable | Description |
|---|---|
| `cnt` | **Target**  total daily bike rentals |
| `temp` | Normalised temperature (multiply by 41 for °C) |
| `hum` | Normalised humidity |
| `windspeed` | Normalised wind speed |
| `season` | 1=Winter, 2=Spring, 3=Summer, 4=Fall |
| `weathersit` | 1=Clear, 2=Mist, 3=Light Rain, 4=Heavy Rain |
| `workingday` | 1=Working day, 0=Weekend/holiday |
| `yr` | 0=2011, 1=2012 |

---

## Why Not Linear Regression?

```
Mean of cnt:          4504.3
Variance of cnt:   3,752,788
Variance/Mean ratio:    833.2
```

The variance is **833× the mean**  massive overdispersion. Linear regression assumes constant variance (homoskedasticity), which is violated. The response is also a count (non-negative integer), so:

- **Step 1:** Try Poisson GLM (assumes mean = variance)
- **Step 2:** Test for overdispersion (Pearson χ²/df >> 1)
- **Step 3:** Switch to Negative Binomial (adds a dispersion parameter)

---

## Methods

### R (`bikeshare_analysis.R`)
| Step | Function | Package |
|---|---|---|
| Overdispersion test | `dispersiontest()` | AER |
| Poisson GLM | `glm(family = poisson)` | base R |
| Negative Binomial | `glm.nb()` | MASS |
| IRRs with CI | `tidy(exponentiate = TRUE)` | broom |
| Model comparison | `AIC()` | base R |
| Type II LRT | `Anova(type="II")` | car |

### Python (`bikeshare_analysis.py`)
| Step | Function | Package |
|---|---|---|
| Poisson GLM | `smf.glm(family=Poisson())` | statsmodels |
| Pearson χ²/df | `model.pearson_chi2 / df_resid` | statsmodels |
| Negative Binomial | `sm.NegativeBinomial().fit()` | statsmodels |
| IRRs | `np.exp(model.params)` | numpy |
| AIC comparison | `model.aic` | statsmodels |

---

## Key Results

### Overdispersion
| | Value |
|---|---|
| Mean (cnt) | 4,504 |
| Variance (cnt) | 3,752,788 |
| Variance/Mean | **833** → strongly overdispersed |
| Pearson χ²/df (Poisson) | **174.2** (720 residual df) |

### Model Comparison
| Model | AIC |
|---|---|
| Poisson GLM | 142,141 |
| **Negative Binomial** | **12,199** ✅ |

→ Negative Binomial is substantially better (lower AIC).

### Incidence Rate Ratios (Negative Binomial)

Values come from the Python run (`bikeshare_analysis.py`, statsmodels, NB dispersion alpha = 0.0617). The baseline season is **Winter** (season 1). The R script was not re-run, so no R numbers are reported here.

| Predictor | IRR | Interpretation |
|---|---|---|
| Year 2012 vs 2011 | 1.62 | 62% more rentals, system grew |
| Fall vs Winter | 1.62 | 62% more rentals in Fall |
| Spring vs Winter | 1.39 | 39% more |
| Summer vs Winter | 1.25 | 25% more |
| Temperature (+1 unit, normalised scale) | 4.48 | Higher temp = many more rentals; +1 unit spans the whole observed range |
| Light Rain vs Clear | 0.49 | 51% fewer rentals |
| Humidity (continuous, +1 unit) | 0.73 | Coefficient -0.321; +1 unit spans the whole observed range |
| Humidity (continuous, +0.1 unit) | 0.97 | About 3% fewer rentals per 0.1 increase |
| Working day | 1.06 | Slight increase vs weekends |

### 6-Step Hypothesis Test: Season Effect
1. **α = 0.05**
2. **H₀:** No season effect on log rental rate
3. **Decision rule:** Reject H₀ if p-value ≤ 0.05
4. **χ²(3) = 323.4** from a likelihood-ratio test (NB GLM with and without season, α fixed at 0.0617 from the full model; re-run in Python on `day.csv`)
5. **p < 0.001** (about 9e-70)
6. **Conclusion:** Season significantly affects daily rental count

The R script was not re-run for this figure; its Type II test may differ slightly.

---

## How to Reproduce

### Download the data
1. Go to: https://archive.uci.edu/dataset/275/bike+sharing+dataset
2. Download the ZIP → extract `day.csv` and `hour.csv`
3. Place both files in a `data/` folder next to the scripts (`data/day.csv`, `data/hour.csv`); create a `figures/` folder too, since the Python script saves plots there

The data file is not included in this repo. It is the UCI Bike Sharing Dataset (`day.csv`, 731 rows).

### R
```r
install.packages(c("tidyverse","MASS","broom","car","AER"))
source("bikeshare_analysis.R")
```

### Python
```bash
pip install pandas numpy matplotlib seaborn statsmodels
python bikeshare_analysis.py
```

The numbers in Key Results were measured by running the Python script's model sections (Poisson, Negative Binomial, IRRs) on `day.csv`. The R script was not re-run for this update, so its output may differ slightly.

---

## What I'd Do Next
- Model **hourly** demand with the 17,379-row `hour.csv`  more granular and more complex
- Add a **time series component** (Prophet or SARIMA) to capture trends over the two years
- Try a **Random Forest** or **Gradient Boosting** regressor and compare RMSE vs the GLM
- Build a **rental demand calculator**: input weather forecast → predict tomorrow's demand

---

*Data: Fanaee-T, H. & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. Progress in AI, 2(2-3), 113–127. DOI: 10.24432/C5W894. License: CC BY 4.0.*
