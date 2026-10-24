# delhivery-delay-intelligence

## Data findings

1. **Service type:** Express has the highest delay rate, at 0.4897 (48.97%).

2. **Hub tier:** The delay rate rises from 0.0400 at Tier 1 (16 of 400 shipments), to 0.2150 at Tier 2 (86 of 400), and 0.4400 at Tier 3 (176 of 400).

3. **Distance band:** Delay rates rise with distance: 0.0219 for 0–200 km (6 of 274 shipments), 0.1368 for 201–400 km (77 of 563), 0.4364 for 401–600 km (120 of 275), and 0.8523 for 601+ km (75 of 88).

4. **Correlations:** The three strongest correlations with `delivery_hours` are `promised_hours` (0.8372), `distance_km` (0.4013), and `n_stops` (0.3247).

5. **Declared-value transformation:** The filled `declared_value_inr` values are right-skewed, with a mean of INR 1,533.0542 well above the median of INR 1,121.50; after taking the natural log, the mean (7.0277) and median (7.0224) nearly meet, showing how the transformation compresses the long tail of high values.