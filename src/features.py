from sklearn.preprocessing import StandardScaler
from src.config import FEATURE_PARAMS
import joblib
import os

# Fixed mappings
GEOGRAPHY_MAP = {"France": 0, "Germany": 1, "Spain": 2}
GENDER_MAP = {"Male": 0, "Female": 1}


def preprocess(X_train, X_test):
    # Drop useless columns
    drop_cols = FEATURE_PARAMS["drop_columns"]
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.drop(columns=drop_cols, errors="ignore")

    # Encode categorical b fixed mapping
    X_train["Geography"] = X_train["Geography"].map(GEOGRAPHY_MAP)
    X_test["Geography"] = X_test["Geography"].map(GEOGRAPHY_MAP)
    X_train["Gender"] = X_train["Gender"].map(GENDER_MAP)
    X_test["Gender"] = X_test["Gender"].map(GENDER_MAP)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler only
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    joblib.dump(
        {"geography": GEOGRAPHY_MAP, "gender": GENDER_MAP}, "models/encoder.pkl"
    )

    return X_train_scaled, X_test_scaled


if __name__ == "__main__":
    import numpy as np
    from src.data import load_data, split_data

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_train.npy", X_train_scaled)
    np.save("data/processed/X_test.npy", X_test_scaled)
    np.save("data/processed/y_train.npy", y_train.values)
    np.save("data/processed/y_test.npy", y_test.values)
    print("✅ Preprocessing done! Saved to data/processed/")
