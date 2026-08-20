"""
Tests unitaires des fonctions de feature engineering.

Ces tests portent uniquement sur les fonctions pures de transformations.py
(pas de dépendance à MLflow ni au dataset réel), pour rester rapides et
faciles à lancer en local (`pytest tests/`).
"""

import pandas as pd

from src.features.transformations import (
    drop_id_columns,
    impute_numeric,
    impute_categorical,
    encode_categorical,
)


def make_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Transaction_ID": ["T1", "T2"],
            "User_ID": [1, 2],
            "Transaction_Amount": [100.0, None],
            "Transaction_Type": ["ATM Withdrawal", "Bill Payment"],
            "Time_of_Transaction": [10.0, 12.0],
            "Device_Used": ["Mobile", None],
            "Location": ["New York", "Los Angeles"],
            "Previous_Fraudulent_Transactions": [0, 1],
            "Account_Age": [10, 20],
            "Number_of_Transactions_Last_24H": [1, 2],
            "Payment_Method": ["UPI", None],
        }
    )


def test_drop_id_columns_removes_identifiers():
    df = make_sample_df()
    result = drop_id_columns(df)
    assert "Transaction_ID" not in result.columns
    assert "User_ID" not in result.columns
    # Les autres colonnes doivent rester intactes.
    assert "Transaction_Amount" in result.columns


def test_impute_numeric_fills_missing_with_median():
    df = make_sample_df()
    result, medians = impute_numeric(df)
    assert result["Transaction_Amount"].isna().sum() == 0
    # La valeur manquante doit être remplacée par la médiane calculée.
    assert result["Transaction_Amount"].iloc[1] == medians["Transaction_Amount"]


def test_impute_numeric_reuses_given_medians():
    """Vérifie qu'on peut appliquer des médianes précalculées (cas du test set)."""
    df = make_sample_df()
    fixed_medians = pd.Series({col: 999.0 for col in df.columns if df[col].dtype != object})
    result, medians_used = impute_numeric(df, medians=fixed_medians)
    assert medians_used is fixed_medians
    assert result["Transaction_Amount"].iloc[1] == 999.0


def test_impute_categorical_fills_unknown():
    df = make_sample_df()
    result = impute_categorical(df)
    assert result["Device_Used"].isna().sum() == 0
    assert result["Payment_Method"].isna().sum() == 0
    assert (result["Device_Used"] == "Unknown").any()


def test_encode_categorical_creates_dummy_columns():
    df = impute_categorical(make_sample_df())
    encoded = encode_categorical(df)
    assert any(col.startswith("Device_Used_") for col in encoded.columns)
    assert any(col.startswith("Payment_Method_") for col in encoded.columns)


def test_encode_categorical_aligns_to_reference_columns():
    """
    Vérifie que le test set peut être encodé avec exactement les mêmes
    colonnes que le train, même si une catégorie n'apparaît que dans l'un
    des deux ensembles (évite un mismatch de colonnes à l'entraînement).
    """
    train_df = impute_categorical(make_sample_df())
    train_encoded = encode_categorical(train_df)

    # Un test set avec une catégorie absente du train (ex: "Tablet").
    test_df = make_sample_df()
    test_df.loc[0, "Device_Used"] = "Tablet"
    test_df = impute_categorical(test_df)
    test_encoded = encode_categorical(test_df, reference_columns=train_encoded.columns)

    assert list(test_encoded.columns) == list(train_encoded.columns)
