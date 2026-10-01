import pandas as pd

from model import predict_quantity, train_model


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "produto": list("ABCDEFGH"),
            "categoria": ["Casa"] * 8,
            "preco": [10, 20, 30, 40, 50, 60, 70, 80],
            "quantidade_vendida": [80, 70, 60, 50, 40, 30, 20, 10],
        }
    )


def test_train_model_returns_metrics():
    coefficients, metrics = train_model(sample_data())
    assert len(coefficients) == 3
    assert set(metrics) == {"mae", "baseline_mae", "r2"}


def test_predict_quantity_is_non_negative():
    prediction = predict_quantity(sample_data(), 25, "Casa")
    assert isinstance(prediction, float)
    assert prediction >= 0
