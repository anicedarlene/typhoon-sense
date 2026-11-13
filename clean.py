
import re
import glob
import os
import numpy as np
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================

COLS = ["year", "month", "day", "value", "flag"]

# ==========================================
# 2. READ HKO CSV FILES
# ==========================================

def read_hko_csv(path):

    for enc in ("utf-8-sig", "big5hkscs", "utf-8"):

        try:
            with open(path, encoding=enc) as f:
                titles = [next(f) for _ in range(2)]

            df = pd.read_csv(
                path,
                skiprows=3,
                header=None,
                names=COLS,
                usecols=range(5),
                dtype=str,
                encoding=enc,
                engine="python",
                on_bad_lines="skip"
            )

            return df, titles

        except UnicodeDecodeError:
            continue

    raise ValueError(f"Cannot read file: {path}")


# ==========================================
# 3. GENERAL CLEANING FUNCTION
# ==========================================

def clean(df, value_name):

    for c in ["year", "month", "day"]:
        df[c] = pd.to_numeric(
            df[c], errors="coerce"
        )

    df = df.dropna(
        subset=["year", "month", "day"]
    ).copy()

    df["date"] = pd.to_datetime(
        dict(
            year=df.year.astype(int),
            month=df.month.astype(int),
            day=df.day.astype(int)
        ),
        errors="coerce"
    )

    df = df.dropna(subset=["date"])

    df[value_name] = pd.to_numeric(
        df["value"].str.strip(),
        errors="coerce"
    )

    df["flag"] = df["flag"].str.strip()

    return df[["date", value_name, "flag"]]


# ==========================================
# 4. CLEAN WIND DATA
# ==========================================

frames = []

wind_files = sorted(
    glob.glob("data/raw/wind/daily_*_WSPD_*.csv")
)

print("Wind CSV files found:", len(wind_files))

if len(wind_files) == 0:
    raise FileNotFoundError("No wind CSV files found.")

for path in wind_files:

    raw, titles = read_hko_csv(path)

    d = clean(raw, "wind_kmh")

    station = re.search(
        r"daily_(.+)_WSPD",
        os.path.basename(path)
    ).group(1)

    d["station"] = station

    frames.append(d)

wind = pd.concat(
    frames,
    ignore_index=True
)

# Remove duplicate station-date records
wind = wind.drop_duplicates(
    subset=["station", "date"]
)

# Remove negative wind speed values
wind.loc[
    wind["wind_kmh"] < 0,
    "wind_kmh"
] = np.nan

# ==========================================
# 5. CLEAN HUMIDITY DATA
# ==========================================

raw, titles = read_hko_csv(
    "data/raw/humidity.csv"
)

humidity = clean(
    raw,
    "rh_pct"
)

humidity = humidity.drop_duplicates(
    subset=["date"]
)

# Humidity must be between 0 and 100
bad_humidity = (
    (humidity["rh_pct"] < 0) |
    (humidity["rh_pct"] > 100)
)

humidity.loc[
    bad_humidity,
    "rh_pct"
] = np.nan

# ==========================================
# 6. SAVE CLEANED RAW DATA
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

wind.to_csv(
    "data/processed/cleaned_wind.csv",
    index=False
)

humidity.to_csv(
    "data/processed/cleaned_humidity.csv",
    index=False
)

print("\nCleaning completed!")
print("Wind records:", len(wind))
print("Wind stations:", wind["station"].nunique())
print("Humidity records:", len(humidity))
print("Missing wind values:", wind["wind_kmh"].isna().sum())
print("Missing humidity values:", humidity["rh_pct"].isna().sum())
