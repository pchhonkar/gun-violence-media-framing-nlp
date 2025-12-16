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
- χ² = 5.6910
- df = 3
- p-value = 0.127651
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
- χ² = 0.1462
- df = 3
- p-value = 0.985765
- **Decision**: Fail to reject H₀ (α = 0.05)

**Interpretation**: No statistically significant difference in 'Action descriptions' usage across outlets. The observed variation could be due to chance.

**Residual Analysis**:
- Overuse: **WSJ** (higher than expected)
- Underuse: **Fox** (lower than expected)

---

### Test 3: Harm severity

**Null Hypothesis (H₀)**: The proportion of descriptions classified as "Harm severity" 
is the same across all four outlets (CNN, Fox, NYT, WSJ).

**Alternative Hypothesis (H₁)**: At least one outlet has a different proportion.

**Results**:
- χ² = 0.8545
- df = 3
- p-value = 0.836396
- **Decision**: Fail to reject H₀ (α = 0.05)

**Interpretation**: No statistically significant difference in 'Harm severity' usage across outlets. The observed variation could be due to chance.

**Residual Analysis**:
- Overuse: **NYT** (higher than expected)
- Underuse: **Fox** (lower than expected)

---

## Shooter Cluster Tests

### Test 4: Shooter identity labels

**Null Hypothesis (H₀)**: The proportion of descriptions classified as "Shooter identity labels" 
is the same across all four outlets (CNN, Fox, NYT, WSJ).

**Alternative Hypothesis (H₁)**: At least one outlet has a different proportion.

**Results**:
- χ² = 23.4280
- df = 3
- p-value = 0.000033
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
- χ² = 12.5820
- df = 3
- p-value = 0.005633
- **Decision**: Reject H₀ (α = 0.05)

**Interpretation**: There is a statistically significant difference in how outlets use 'Harm severity' framing. Fox uses this framing more than expected, while NYT uses it less than expected.

**Residual Analysis**:
- Overuse: **Fox** (higher than expected)
- Underuse: **NYT** (lower than expected)

---

### Test 6: Active threat framing

**Null Hypothesis (H₀)**: The proportion of descriptions classified as "Active threat framing" 
is the same across all four outlets (CNN, Fox, NYT, WSJ).

**Alternative Hypothesis (H₁)**: At least one outlet has a different proportion.

**Results**:
- χ² = 4.4415
- df = 3
- p-value = 0.217569
- **Decision**: Fail to reject H₀ (α = 0.05)

**Interpretation**: No statistically significant difference in 'Active threat framing' usage across outlets. The observed variation could be due to chance.

**Residual Analysis**:
- Overuse: **WSJ** (higher than expected)
- Underuse: **NYT** (lower than expected)

---

## Summary

| Entity Type | Cluster | χ² | p-value | Reject H₀? | Overuse | Underuse |
|-------------|---------|-----|---------|------------|---------|----------|
| victim | Victim age & count framing... | 5.69 | 0.1277 | No | WSJ | Fox |
| victim | Action descriptions... | 0.15 | 0.9858 | No | WSJ | Fox |
| victim | Harm severity... | 0.85 | 0.8364 | No | NYT | Fox |
| shooter | Shooter identity labels... | 23.43 | 0.0000 | Yes | CNN | Fox |
| shooter | Harm severity... | 12.58 | 0.0056 | Yes | Fox | NYT |
| shooter | Active threat framing... | 4.44 | 0.2176 | No | WSJ | NYT |

**Overall**: 2 out of 6 tests showed statistically significant differences at α = 0.05.

## Conclusion

The Chi-Square tests reveal significant heterogeneity in how different outlets frame victims and shooters. 
The residual analysis helps identify which outlets drive these differences, showing patterns of over-use or under-use relative to expected frequencies under the null hypothesis.
