# delhivery-delay-intelligence

## Data findings

1. **Service type:** Express has the highest delay rate, at 0.4897 (48.97%).

2. **Hub tier:** The delay rate rises from 0.0400 at Tier 1 (16 of 400 shipments), to 0.2150 at Tier 2 (86 of 400), and 0.4400 at Tier 3 (176 of 400).

3. **Distance band:** Delay rates rise with distance: 0.0219 for 0–200 km (6 of 274 shipments), 0.1368 for 201–400 km (77 of 563), 0.4364 for 401–600 km (120 of 275), and 0.8523 for 601+ km (75 of 88).

4. **Correlations:** The three strongest correlations with `delivery_hours` are `promised_hours` (0.8372), `distance_km` (0.4013), and `n_stops` (0.3247).

5. **Declared-value transformation:** The filled `declared_value_inr` values are right-skewed, with a mean of INR 1,533.0542 well above the median of INR 1,121.50; after taking the natural log, the mean (7.0277) and median (7.0224) nearly meet, showing how the transformation compresses the long tail of high values.

## Model results (regression and classification)
## Regression

1. **Baseline:** The mean baseline had an R² of −0.0212. Predicting the training average for every shipment performed slightly worse than predicting the validation average, showing that this simple approach underfits because it ignores differences between shipments.

2. **Validation winner:** Linear regression had the lowest validation RMSE at 4.8912 hours.

3. **Tree depth:** Depths 1–3 show high bias: both training and validation RMSE are high, with small gaps of about 0.50–1.17. Among the tested decision trees, depth 6 has the lowest validation RMSE at 6.2796. The gap at depth 8 (3.45) marks the start of overfitting; it widens further at depth 12 (5.99) and unlimited depth (7.84), showing high variance. The unlimited tree has zero training RMSE but higher validation RMSE, so it fits the training data closely and generalizes less well.


4. **Learning rate:** A learning rate of 0.1 performed best, with a validation R² of 0.9306. At 0.01, the model learned too slowly with 100 trees (R² 0.7127); at 1.0, it was too aggressive and performed worse (R² 0.8440).

5. **Model choice:** Cross-validation modestly favored gradient boosting, with a mean R² of 0.9353 versus 0.9261 for linear regression—a difference of 0.0092, about twice either model’s fold-to-fold standard deviation. I would still choose linear regression because it had the lower validation RMSE (4.8912 versus 4.9165 hours) and is simpler, while recognizing that gradient boosting may perform slightly better across different data splits.

## Classification

1. **Blind Baseline Model:** The blind baseline achieved 0.7667 accuracy by predicting every parcel would be on time, but its recall was 0.0000 because it missed all 42 delayed parcels; it is therefore useless for proactive delay alerts.

2. **Noise Columns:**  The five noise columns account for a combined 0.2245 of the forest’s feature importance, but all rank below the top seven by F-score. Their F-score ranks are 8, 12, 14, 16 and 17.

3. **Number of features:** The table shows that selecting 7 features gave the highest validation F1 (0.8333). The scores for k=5, 10, 12, and 17 were 0.6988, 0.7857, 0.8049, and 0.7848, respectively.

4. **Tuning:** The untuned forest using the same seven features scored a higher validation F1 (0.8333) than grid search (0.7750) or randomized search (0.7848). Tuning did not improve F1 in this run, but with only 180 validation parcels, a few cases can affect the comparison; this result doesn’t show that tuning is always unhelpful.

5. **Alert threshold:** The chosen threshold was 0.5, with validation precision of 0.7955 and recall of 0.8333. It caught 35 of 42 delayed parcels, with 9 false alerts and 7 missed delays. At 0.6, precision rose to 0.8889, but recall fell to 0.7619—below the required 0.80.

6. **Final test:** The model had test precision of 0.6889, recall of 0.7561, and F1 of 0.7209. Its test recall was below validation recall (0.8333): it caught 31 of the 41 delayed test parcels and missed 10. Because the test set has only 41 delayed parcels, a few cases can noticeably affect the recall.

## How to run

The project was built with Python 3.12.9. The pinned library versions install from prebuilt packages on Python 3.12; installation failed on Python 3.14.

**Clone the repository and enter its folder:**
```
git clone https://github.com/tyaginishant0071-hash/delhivery-delay-intelligence.git
cd delhivery-delay-intelligence
```

**Windows PowerShell:** 
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this command and try activating again:
```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**macOS / Linux:**
```
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Run the project in this order:**
```
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

**package_model.py:** models/delay_model.pkl is regenerated by running python package_model.py. The committed pickle loads reliably only with the scikit-learn version pinned in requirements.txt. If installed versions differ, the grader reruns package_model.py instead of loading the committed pickle.

## Operations recommendation

Nine hubs are in the Escalate segment. Their average hub-level delay rate is 0.46 (46%), and their mean delivery time is 64.65 hours:
- HUB13 — Nagpur, Tier 2
- HUB15 — Kochi, Tier 2
- HUB17 — Patna, Tier 3
- HUB18 — Ranchi, Tier 3
- HUB19 — Guwahati, Tier 3
- HUB20 — Raipur, Tier 3
- HUB21 — Bhubaneswar, Tier 3
- HUB23 — Siliguri, Tier 3
- HUB24 — Agartala, Tier 3

Prioritize these hubs for hub-manager review. The segment also has higher average transit distances and stop counts. Managers can investigate whether routes, stops, or local operating conditions contribute to the delays. These patterns suggest where to investigate; they don’t prove the cause.

**Tier and segment mismatches:** Four hubs’ tier labels imply a different playbook from their behavior-based segment: HUB07 (Kolkata, Tier 1) is Watch; HUB13 (Nagpur, Tier 2) and HUB15 (Kochi, Tier 2) are Escalate; and HUB22 (Dehradun, Tier 3) is Watch. For flagged parcel SHP90005 at HUB22, using the tier alone would imply manager escalation because Tier 3 maps to Escalate. Its Watch segment routes it to a proactive customer call. Use the observed segment alongside the tier label when choosing a hub response.

**Alert threshold:** A parcel is flagged when its predicted delay probability is at least 0.5. This was the highest tested threshold that kept validation recall at or above 0.80: recall was 0.8333 at 0.5 and fell to 0.7619 at 0.6. The chosen threshold favors catching more parcels that may be late, even though some on-time parcels may receive an unnecessary call. This trade-off reflects the higher cost of missing a delayed parcel.

**Test alert results:** Precision was 0.6889, meaning about 7 of every 10 flagged parcels were really delayed. Recall was 0.7561, meaning about 7 to 8 of every 10 delayed parcels were caught. Test recall was lower than validation recall (0.8333). The test set had only 41 delayed parcels, so each parcel changes recall by about 0.024, or 2.4 percentage points.

Of the 48 new parcels, 8 were flagged. 6 require escalation to a hub manager, and the other 2 require proactive customer calls.

I kept K = 3 to match the three operational playbooks: Standard, Watch, and Escalate. The separation ratio rises from 5.3928 at K = 3 to 6.3047 at K = 4 and 6.8866 at K = 5, but the additional clusters become small: K = 4 has a cluster of four hubs, and K = 5 has clusters of three and two hubs. Those small groups would be harder to use as reliable, separate playbooks.

## Production plan

### Five characteristics

**Accuracy:** On the validation rows, the final classifier achieved 0.9111 accuracy at threshold 0.5, compared with 0.7667 for the blind baseline. The baseline caught none of the delayed parcels, so its recall was 0.0000. This shows why accuracy alone can be misleading and why recall matters for proactive delay alerts.

**Reliability:** outputs/part_d_reload_check.csv checks that the saved model produces the same delay probabilities after it is reloaded.

**Reproducibility:** The scripts use fixed random seeds and are listed in run order in this README. Running them with the same data and pinned library versions should reproduce the results.

**Maintainability:** The project separates work into scripts for each stage. Shared preparation functions are in src/shipment_prep.py, making it easier to update cleaning and feature-building steps.

**Consistency:** Training and prediction use the same `clean_shipments` and `build_features` functions, helping prepare new parcels in the same way as the training data.

**Monitorability:** `outputs/new_shipment_actions.csv` records each parcel’s delay probability, flag, hub segment, and recommended action.


### Monitoring quantities and triggers

These are proposed starting triggers, anchored to this project’s results. They should be reviewed with the operations team as real production data becomes available.

**Missing values:** Before cleaning, the training data had missing values in 3.0% of distance_km, 2.0% of parcel_weight_kg (about 2.4% when the five scanner-fault weights are included), and 1.0% of declared_value. Investigate if more than 5% of incoming values are missing in any of these fields. Use the saved training medians from models/impute_values.csv to fill missing values.

**Incoming feature medians:** The saved training medians are 314.95 km for distance, 1.83 kg for parcel weight, and INR 1,121.50 for declared value. Before imputation, investigate if an incoming median differs from its saved training median by more than 20%.

**Share of parcels flagged:** At the chosen threshold, 44 of 180 validation parcels (24.4%) and 45 of 180 test parcels (25.0%) were flagged. Investigate if more than 35% of parcels are flagged in a week.

**Delayed share:** The training delay rate was 23.17%. Monitor the share of late parcels over a rolling four-week window and investigate if it exceeds 30%. A four-week window gives a more stable reading than one week, which has about 93 parcels and a typical fluctuation of roughly 4.4 percentage points. Compare sale-period results with similar sale periods.

**Recall after outcomes are known:** Investigate if recall falls below 0.70 over a sufficiently large review period. This is a proposed early-warning level: test recall was 0.7561, and the brief’s lower bound is 0.55. The test set contained only 41 delayed parcels, so assess recall over a larger rolling group rather than reacting to a small batch.

### Scoring rule check

**Check that 100% of parcels follow the action rules:** flagged parcels at Escalate hubs go to a hub manager; other flagged parcels receive a proactive customer call; unflagged parcels receive no action. Investigate any mismatch.

### Sale periods and data drift

Parcels from the 15–28 September sale window may behave differently from parcels in ordinary weeks. Identify sale parcels using sale_event_flag and compare them with parcels from a similar sale period.
Data drift means new parcels or operating conditions start to differ from the data used to train the model. In this project, it could appear as changes in delivery distances, service types, hub capacities, or how often sale_event_flag is set. It could also appear as a change in delay rates after delivery outcomes are known, or in the share of parcels flagged. Investigate these changes to see whether the model still reflects current operations.

### Retraining schedule

Review monitoring results monthly and retrain quarterly using recent parcels whose delivery outcomes are known. Consider retraining sooner if a trigger is repeatedly exceeded and an investigation confirms an operating or data change. For each retraining run, repeat validation and select the highest alert threshold that keeps validation recall at or above 0.80. Evaluate the updated model on separate test data before using it for parcel actions.

## Limitations
**Synthetic data:** The model was developed using 1,200 generated shipments from 24 hubs over a 90-day period. Real parcel mix, routes, hub operations, and delays may differ, so these results need validation on real operational data before being relied on in practice.

**Small test set:** The test set had only 41 delayed parcels. Test recall was 0.7561, compared with validation recall of 0.8333. One additional parcel caught or missed changes test recall by about 0.024, so the test result is uncertain.

**Threshold selected on one validation split:** The alert threshold was selected using 180 validation parcels, including 42 delayed parcels. Recall was 0.8333 at threshold 0.5, but 0.7619 at 0.6. The chosen threshold may not have the same precision and recall trade-off on future parcels.

**Model selection is uncertain:** The untuned forest scored validation F1 of 0.8333, compared with 0.7750 for grid search and 0.7848 for randomized search on the same 180-row validation set. A small validation sample may not reliably identify which setup will perform best on future data.