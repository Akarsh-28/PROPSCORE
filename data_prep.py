"""
Load the raw CSV and turn it into a clean (X, y) ready for modeling.
"""

import pandas as pd
import config


def load_raw_data():
    """Read the raw CSV into a DataFrame."""
    return pd.read_csv(config.INPUT_CSV)


def build_features(df):
    """
    Turn the raw DataFrame into:
      X - feature matrix (leakage-free, encoded, no missing values)
      y - target (0/1, did the user convert)
    """
    y = df[config.TARGET_COL].astype(bool).astype(int)

    drop_cols = config.LEAKAGE_COLS + config.ID_COLS
    X = df.drop(columns=drop_cols)

    # Missing feature-usage flags mean "not applicable to this user_type" -> 0
    for col in config.BOOL_FLAG_COLS:
        X[col] = (
            X[col]
            .map({True: 1, False: 0, "True": 1, "False": 0})
            .fillna(0)
            .astype(int)
        )

    # One-hot encode text columns
    X = pd.get_dummies(X, columns=config.CATEGORICAL_COLS, drop_first=False)

    return X, y


def load_features():
    """Convenience wrapper: raw CSV -> (df, X, y) in one call."""
    df = load_raw_data()
    X, y = build_features(df)
    return df, X, y