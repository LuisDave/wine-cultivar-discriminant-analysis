# Methodology

The project studies whether 13 chemical measurements can be compressed into a small supervised representation while retaining separation among three wine cultivars.

1. Preserve and audit the raw Kaggle CSV.
2. Assign canonical variable names because the source file has no header.
3. Validate numeric types, missing values, duplicates and class balance.
4. Standardize the 13 predictors.
5. Fit Linear Discriminant Analysis (LDA). With three classes, the discriminant space has at most `K - 1 = 2` dimensions.
6. Measure the relative discriminant variance of LD1 and LD2.
7. Inspect the standardized discriminant scalings to identify variables most involved in class separation.
8. Evaluate predictive performance on a stratified holdout set and with stratified cross-validation.

The dimensionality-reduction analysis is fitted on the full clean dataset for descriptive interpretation; predictive metrics are computed separately using resampling to avoid treating in-sample separation as out-of-sample accuracy.
