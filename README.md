# delhivery-delay-intelligence

## Data findings

1. **Service type:** Express has the highest delay rate, at 0.4897 (48.97%).

2. **Hub tier:** The delay rate rises from 0.0400 at Tier 1 (16 of 400 shipments), to 0.2150 at Tier 2 (86 of 400), and 0.4400 at Tier 3 (176 of 400).

3. **Distance band:** Delay rates rise with distance: 0.0219 for 0–200 km (6 of 274 shipments), 0.1368 for 201–400 km (77 of 563), 0.4364 for 401–600 km (120 of 275), and 0.8523 for 601+ km (75 of 88).

4. **Correlations:** The three strongest correlations with `delivery_hours` are `promised_hours` (0.8372), `distance_km` (0.4013), and `n_stops` (0.3247).

5. **Declared-value transformation:** The filled `declared_value_inr` values are right-skewed, with a mean of INR 1,533.0542 well above the median of INR 1,121.50; after taking the natural log, the mean (7.0277) and median (7.0224) nearly meet, showing how the transformation compresses the long tail of high values.

## Model Results (regression and classification)
### Regression

1. **Baseline:** The mean baseline had an R² of −0.0212. Predicting the training average for every shipment performed slightly worse than predicting the validation average, showing that this simple approach underfits because it ignores differences between shipments.

2. **Validation winner:** Linear regression had the lowest validation RMSE at 4.8912 hours.

3. **Tree depth:** Depths 1–3 show high bias: both training and validation RMSE are high, with small gaps of about 0.50–1.17. Among the tested decision trees, depth 6 has the lowest validation RMSE at 6.2796. The gap at depth 8 (3.45) marks the start of overfitting; it widens further at depth 12 (5.99) and unlimited depth (7.84), showing high variance. The unlimited tree has zero training RMSE but higher validation RMSE, so it fits the training data closely and generalizes less well.


4. **Learning rate:** A learning rate of 0.1 performed best, with a validation R² of 0.9306. At 0.01, the model learned too slowly with 100 trees (R² 0.7127); at 1.0, it was too aggressive and performed worse (R² 0.8440).

5. **Model choice:** Cross-validation modestly favored gradient boosting, with a mean R² of 0.9353 versus 0.9261 for linear regression—a difference of 0.0092, about twice either model’s fold-to-fold standard deviation. I would still choose linear regression because it had the lower validation RMSE (4.8912 versus 4.9165 hours) and is simpler, while recognizing that gradient boosting may perform slightly better across different data splits.

## Classification

1. **Blind Baseline Model:** The blind baseline achieved 0.7667 accuracy by predicting every parcel would be on time, but its recall was 0.0000 because it missed all 42 delayed parcels; it is therefore useless for proactive delay alerts.

2. **Noise Columns:**  The five noise columns account for a combined 0.2245 of the forest’s feature importance, but all rank below the top seven by F-score. Their F-scores rank are: 8, 12, 14, 16, and 17.

3. **Number of features:** The table shows that selecting 7 features gave the highest validation F1 (0.8333). The scores for k=5, 10, 12, and 17 were 0.6988, 0.7857, 0.8049, and 0.7848, respectively.

4. **Tuning:** The untuned forest using the same seven features scored a higher validation F1 (0.8333) than grid search (0.7750) or randomized search (0.7848). Tuning did not improve F1 in this run, but with only 180 validation parcels, a few cases can affect the comparison; this result doesn’t show that tuning is always unhelpful.

5. **Alert threshold:** The chosen threshold was 0.5, with validation precision of 0.7955 and recall of 0.8333. It caught 35 of 42 delayed parcels, with 9 false alerts and 7 missed delays. At 0.6, precision rose to 0.8889, but recall fell to 0.7619—below the required 0.80.

6. **Final test:** The model had test precision of 0.6889, recall of 0.7561, and F1 of 0.7209. Its test recall was below validation recall (0.8333): it caught 31 of the 41 delayed test parcels and missed 10. Because the test set has only 41 delayed parcels, a few cases can noticeably affect the recall.

