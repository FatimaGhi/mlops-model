import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_PATH, DATA_PARAMS


def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def split_data(df):
    print(df.columns.tolist())
    target = DATA_PARAMS["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=DATA_PARAMS["test_size"],
        random_state=DATA_PARAMS["random_state"],
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
