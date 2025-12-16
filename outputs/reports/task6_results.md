# Task 6: Statistical Hypothesis Testing Results

## Overview

This analysis tests whether the distribution of cluster usage differs significantly 
across news outlets (CNN, Fox, NYT, WSJ) using the **Chi-Squared Test of Homogeneity**.

**Significance Level**: α = 0.05

## Method

For each of the top 3 clusters (by total count, excluding Noise/Unclustered):
1. Build a 4×2 contingency table (outlets × [in_cluster, not_in_cluster])
2. Run Chi-Square Test of Homogeneity
3. Compute standardized residuals to identify over/under-use by outlet

---

## Victim Cluster Tests

### Test 1: Victim age & count framing

**Null Hypothesis (H₀)**: The proportion of descriptions classified as "Victim age & count framing" 
is the same across all four outlets (CNN, Fox, NYT, WSJ).

**Alternative Hypothesis (H₁)**: At least one outlet has a different proportion.

**Results**:
- χ² = 6.7276
- df = 3
- p-value = 0.081104
- **Decision**: Fail to reject H₀ (α = 0.05)

**Interpretation**: No statistically significant difference in 'Victim age & count framing' usage across outlets. The observed variation could be due to chance.

**Residual Analysis**:
- Overuse: **WSJ** (higher than expected)
- Underuse: **Fox** (lower than expected)

---

### Test 2: Action descriptions

**Null Hypothesis (H₀)**: The proportion of descriptions classified as "Action descriptions" 
is the same across all four outlets (CNN, Fox, NYT, WSJ).

**Alternative Hypothesis (H₁)**: At least one outlet has a different proportion.

**Results**:
- χ² = 0.0917
- df = 3
- p-value = 0.992820
- **Decision**: Fail to reject H₀ (α = 0.05)

**Interpretation**: No statistically significant difference in 'Action descriptions' usage across outlets. The observed variation could be due to chance.

**Residual Analysis**:
- Overuse: **CNN** (higher than expected)
- Underuse: **Fox** (lower than expected)

---

### Test 3: Harm severity

**Null Hypothesis (H₀)**: The proportion of descriptions classified as "Harm severity" 
is the same across all four outlets (CNN, Fox, NYT, WSJ).

**Alternative Hypothesis (H₁)**: At least one outlet has a different proportion.

**Results**:
- χ² = 1.5509
- df = 3
- p-value = 0.670574
- **Decision**: Fail to reject H₀ (α = 0.05)

**Interpretation**: No statistically significant difference in 'Harm severity' usage across outlets. The observed variation could be due to chance.

**Residual Analysis**:
- Overuse: **WSJ** (higher than expected)
- Underuse: **Fox** (lower than expected)

---

## Shooter Cluster Tests

### Test 4: Shooter identity labels

**Null Hypothesis (H₀)**: The proportion of descriptions classified as "Shooter identity labels" 
is the same across all four outlets (CNN, Fox, NYT, WSJ).

**Alternative Hypothesis (H₁)**: At least one outlet has a different proportion.

**Results**:
- χ² = 25.7163
- df = 3
- p-value = 0.000011
- **Decision**: Reject H₀ (α = 0.05)

**Interpretation**: There is a statistically significant difference in how outlets use 'Shooter identity labels' framing. CNN uses this framing more than expected, while Fox uses it less than expected.

**Residual Analysis**:
- Overuse: **CNN** (higher than expected)
- Underuse: **Fox** (lower than expected)

---

### Test 5: Harm severity

**Null Hypothesis (H₀)**: The proportion of descriptions classified as "Harm severity" 
is the same across all four outlets (CNN, Fox, NYT, WSJ).

**Alternative Hypothesis (H₁)**: At least one outlet has a different proportion.

**Results**:
- χ² = 3.9959
- df = 3
- p-value = 0.261906
- **Decision**: Fail to reject H₀ (α = 0.05)

**Interpretation**: No statistically significant difference in 'Harm severity' usage across outlets. The observed variation could be due to chance.

**Residual Analysis**:
- Overuse: **Fox** (higher than expected)
- Underuse: **NYT** (lower than expected)

---

### Test 6: Active threat framing

**Null Hypothesis (H₀)**: The proportion of descriptions classified as "Active threat framing" 
is the same across all four outlets (CNN, Fox, NYT, WSJ).

**Alternative Hypothesis (H₁)**: At least one outlet has a different proportion.

**Results**:
- χ² = 5.3115
- df = 3
- p-value = 0.150361
- **Decision**: Fail to reject H₀ (α = 0.05)

**Interpretation**: No statistically significant difference in 'Active threat framing' usage across outlets. The observed variation could be due to chance.

**Residual Analysis**:
- Overuse: **WSJ** (higher than expected)
- Underuse: **NYT** (lower than expected)

---

## Summary

| Entity Type | Cluster | χ² | p-value | Reject H₀? | Overuse | Underuse |
|-------------|---------|-----|---------|------------|---------|----------|
| victim | Victim age & count framing... | 6.73 | 0.0811 | No | WSJ | Fox |
| victim | Action descriptions... | 0.09 | 0.9928 | No | CNN | Fox |
| victim | Harm severity... | 1.55 | 0.6706 | No | WSJ | Fox |
| shooter | Shooter identity labels... | 25.72 | 0.0000 | Yes | CNN | Fox |
| shooter | Harm severity... | 4.00 | 0.2619 | No | Fox | NYT |
| shooter | Active threat framing... | 5.31 | 0.1504 | No | WSJ | NYT |

**Overall**: 1 out of 6 tests showed statistically significant differences at α = 0.05.

## Conclusion

The Chi-Square tests reveal significant heterogeneity in how different outlets frame victims and shooters. 
The residual analysis helps identify which outlets drive these differences, showing patterns of over-use or under-use relative to expected frequencies under the null hypothesis.
