from src.data import load_data, split_data
from src.evaluate import validate_data, check_data_integrity


def test_load_data():
    df = load_data()
    assert df is not None
    assert len(df) > 0
    assert "Exited" in df.columns


def test_split_data():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) > 0
    assert len(X_test) > 0


def test_validate_data():
    df = load_data()
    validate_data(df)


def test_data_integrity():
    df = load_data()
    check_data_integrity(df)
