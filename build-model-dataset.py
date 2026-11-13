import pandas as pd

# 1. Load the processed datasets
weather = pd.read_csv(
    "data/processed/cleaned_weather.csv",
    parse_dates=["date"]
)

cyclone = pd.read_csv(
    "data/processed/daily_cyclone_labels.csv",
    parse_dates=["date"]
)

# 2. Sort both datasets by date
weather = weather.sort_values("date")
cyclone = cyclone.sort_values("date")

# 3. Keep the period from 2015 to 2025
weather = weather[
    (weather["date"] >= "2015-01-01") &
    (weather["date"] <= "2025-12-31")
].copy()

cyclone = cyclone[
    (cyclone["date"] >= "2015-01-01") &
    (cyclone["date"] <= "2025-12-31")
].copy()

# 4. Combine weather data with cyclone labels
dataset = pd.merge(
    weather,
    cyclone,
    on="date",
    how="inner",
    validate="one_to_one"
)

# 5. Check that every calendar day is present
dataset = dataset.sort_values("date").reset_index(drop=True)

date_gaps = dataset["date"].diff().dropna()

if not (date_gaps == pd.Timedelta(days=1)).all():
    raise ValueError(
        "Missing dates detected. Check the weather and cyclone datasets "
        "before creating tomorrow's target."
    )

# 6. Create the target: whether a qualifying cyclone occurs tomorrow
dataset["target_tomorrow"] = (
    dataset["qualifying_cyclone_today"].shift(-1)
)

# 7. Remove the final day because its next-day label is unavailable
dataset = dataset.dropna(subset=["target_tomorrow"]).copy()

dataset["target_tomorrow"] = (
    dataset["target_tomorrow"].astype(int)
)

# 8. Keep the features and target needed for logistic regression
final_dataset = dataset[
    [
        "date",
        "wind_mean",
        "wind_max",
        "n_stations",
        "rh_pct",
        "target_tomorrow"
    ]
].copy()

# 9. Save the final dataset
final_dataset.to_csv(
    "data/processed/model_dataset.csv",
    index=False
)

# 10. Display results
print("Machine learning dataset created!")

print("\nDataset shape:")
print(final_dataset.shape)

print("\nDate range:")
print(final_dataset["date"].min(), "to", final_dataset["date"].max())

print("\nFirst 5 rows:")
print(final_dataset.head().to_string(index=False))

print("\nTarget distribution:")
print(final_dataset["target_tomorrow"].value_counts())

print("\nTarget percentage:")
print(
    final_dataset["target_tomorrow"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)