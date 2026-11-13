import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score
)

# 1. Load the machine learning dataset
df = pd.read_csv(
    "data/processed/model_dataset.csv",
    parse_dates=["date"]
)

# 2. Define features (X) and target (y)
features = [
    "wind_mean",
    "wind_max",
    "n_stations",
    "rh_pct"
]

X = df[features]
y = df["target_tomorrow"]

# 3. Split chronologically: first 80% for training, last 20% for testing
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training period:")
print(df.iloc[0]["date"], "to", df.iloc[split_index - 1]["date"])

print("\nTesting period:")
print(df.iloc[split_index]["date"], "to", df.iloc[-1]["date"])

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# 4. Build the logistic regression pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 7. Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Cyclone", "Cyclone"],
        zero_division=0
    )
)

print("\nAverage Precision (PR-AUC):")
print(round(average_precision_score(y_test, y_prob), 4))