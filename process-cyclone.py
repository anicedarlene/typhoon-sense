
import glob
import os
import numpy as np
import pandas as pd

# ==========================================
# 1. SETTINGS
# ==========================================

INPUT_FOLDER = "data/raw/tropical-cyclone"
OUTPUT_FOLDER = "data/processed"

# Hong Kong Observatory headquarters (reference point)
HK_LAT = 22.302
HK_LON = 114.174

# Research definition
MAX_DISTANCE_KM = 300

# Include Tropical Storm and stronger categories
QUALIFYING_INTENSITIES = {
    "TS",
    "STS",
    "T",
    "ST",
    "SuperT"
}

# HKO CSV column positions
COLUMNS = [
    "name", "year", "month", "day", "time_utc",
    "intensity", "latitude", "longitude",
    "pressure", "max_wind", "jma_code", "hko_code"
]


# ==========================================
# 2. CALCULATE DISTANCE FROM HONG KONG
# ==========================================

def calculate_distance(lat, lon):
    """
    Calculate distance between a cyclone and Hong Kong
    using the Haversine formula.
    """

    earth_radius = 6371.0

    lat1 = np.radians(HK_LAT)
    lon1 = np.radians(HK_LON)
    lat2 = np.radians(lat)
    lon2 = np.radians(lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius * c


# ==========================================
# 3. READ ALL YEARLY CYCLONE FILES
# ==========================================

files = sorted(
    glob.glob(os.path.join(INPUT_FOLDER, "HKO*BST.csv"))
)

if not files:
    raise FileNotFoundError(
        f"No HKO cyclone CSV files found in {INPUT_FOLDER}"
    )

print("Cyclone CSV files found:", len(files))

all_records = []

for file in files:
    print("Reading:", os.path.basename(file))

    df = pd.read_csv(
        file,
        skiprows=3,
        header=None,
        names=COLUMNS,
        encoding="utf-8-sig"
    )

    # Remove empty rows and repeated header rows
    df = df.dropna(how="all")
    df = df[
        pd.to_numeric(df["year"], errors="coerce").notna()
    ].copy()

    all_records.append(df)

cyclone = pd.concat(all_records, ignore_index=True)


# ==========================================
# 4. CLEAN DATE, TIME AND COORDINATES
# ==========================================

numeric_columns = [
    "year", "month", "day", "time_utc",
    "latitude", "longitude"
]

for column in numeric_columns:
    cyclone[column] = pd.to_numeric(
        cyclone[column], errors="coerce"
    )

cyclone = cyclone.dropna(subset=numeric_columns).copy()

# Convert HKO coordinates from hundredths of degrees
cyclone["latitude"] = cyclone["latitude"] / 100
cyclone["longitude"] = cyclone["longitude"] / 100

# Construct UTC timestamp
cyclone["datetime_utc"] = pd.to_datetime(
    dict(
        year=cyclone["year"],
        month=cyclone["month"],
        day=cyclone["day"],
        hour=cyclone["time_utc"]
    ),
    errors="coerce",
    utc=True
)

cyclone = cyclone.dropna(subset=["datetime_utc"]).copy()

# Convert UTC to Hong Kong time (UTC+8)
cyclone["datetime_hk"] = (
    cyclone["datetime_utc"] + pd.Timedelta(hours=8)
)

cyclone["date"] = cyclone["datetime_hk"].dt.normalize()


# ==========================================
# 5. CLEAN INTENSITY
# ==========================================

cyclone["intensity"] = (
    cyclone["intensity"]
    .astype(str)
    .str.strip()
    .str.upper()
)

cyclone["qualifying_intensity"] = (
    cyclone["intensity"].isin(QUALIFYING_INTENSITIES)
)


# ==========================================
# 6. CALCULATE DISTANCE
# ==========================================

cyclone["distance_km"] = calculate_distance(
    cyclone["latitude"],
    cyclone["longitude"]
)

cyclone["within_distance"] = (
    cyclone["distance_km"] <= MAX_DISTANCE_KM
)

# Both conditions must be satisfied
cyclone["qualifying_event"] = (
    cyclone["qualifying_intensity"]
    & cyclone["within_distance"]
)


# ==========================================
# 7. CREATE DAILY YES/NO LABELS
# ==========================================

# Only observations meeting both conditions count
qualifying = cyclone[
    cyclone["qualifying_event"]
].copy()

daily_positive = qualifying.groupby("date").agg(
    qualifying_cyclone_today=("name", lambda x: 1),
    nearest_distance_km=("distance_km", "min"),
    cyclone_names=("name", lambda x: ", ".join(sorted(set(x))))
)


# Create a complete daily calendar for 2015-2025
calendar = pd.date_range(
    start="2015-01-01",
    end="2025-12-31",
    freq="D"
)

# Ensure both indexes use timezone-naive Hong Kong dates
daily_positive.index = (
    pd.DatetimeIndex(daily_positive.index)
    .tz_localize(None)
)

daily_labels = pd.DataFrame(index=calendar)
daily_labels.index.name = "date"

daily_labels = daily_labels.join(daily_positive)


daily_labels["qualifying_cyclone_today"] = (
    daily_labels["qualifying_cyclone_today"]
    .fillna(0)
    .astype(int)
)

daily_labels["nearest_distance_km"] = (
    daily_labels["nearest_distance_km"].round(2)
)

daily_labels["cyclone_names"] = (
    daily_labels["cyclone_names"].fillna("")
)

daily_labels = daily_labels.reset_index()


# ==========================================
# 8. SAVE RESULTS
# ==========================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

output_path = os.path.join(
    OUTPUT_FOLDER,
    "daily_cyclone_labels.csv"
)

daily_labels.to_csv(output_path, index=False)

print("\nCyclone processing completed!")

print("\nTotal cyclone observations:", len(cyclone))

print(
    "Observations meeting both conditions:",
    int(cyclone["qualifying_event"].sum())
)

print(
    "Positive days:",
    int(daily_labels["qualifying_cyclone_today"].sum())
)

print(
    "Negative days:",
    int((daily_labels["qualifying_cyclone_today"] == 0).sum())
)

print("\nFirst positive dates:")
print(
    daily_labels[
        daily_labels["qualifying_cyclone_today"] == 1
    ].head(10).to_string(index=False)
)

print("\nSaved to:", output_path)
