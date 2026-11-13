
import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================

wind = pd.read_csv(
    "data/processed/cleaned_wind.csv",
    parse_dates=["date"]
)

humidity = pd.read_csv(
    "data/processed/cleaned_humidity.csv",
    parse_dates=["date"]
)

# ==========================================
# 2. AGGREGATE WIND STATIONS
# ==========================================

wind_daily = wind.groupby("date").agg(

    wind_mean=("wind_kmh", "mean"),

    wind_max=("wind_kmh", "max"),

    n_stations=("station", "nunique")

)

# ==========================================
# 3. PREPARE HUMIDITY
# ==========================================

humidity_daily = humidity[
    ["date", "rh_pct"]
].set_index("date")

# ==========================================
# 4. MERGE WEATHER DATA
# ==========================================

weather = wind_daily.join(
    humidity_daily,
    how="inner"
)

weather = weather.sort_index()

# ==========================================
# 5. CREATE COMPLETE DATE INDEX
# ==========================================

full_dates = pd.date_range(
    start=weather.index.min(),
    end=weather.index.max(),
    freq="D"
)

weather = weather.reindex(full_dates)

weather.index.name = "date"

# ==========================================
# 6. SAVE WEATHER DATA
# ==========================================

weather.to_csv(
    "data/processed/cleaned_weather.csv"
)

print("Weather dataset created!")
print(weather.head())

print("\nMissing values:")
print(weather.isnull().sum())

print("\nWind station coverage:")
print(weather["n_stations"].describe())
