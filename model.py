from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data/products.csv"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "metrics.json"


def train_model(data: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    """Treina regressão linear com codificação one-hot explícita.

    A implementação usa NumPy para manter o projeto didático e portátil:
    cada categoria vira uma coluna e os coeficientes são ajustados por mínimos
    quadrados. Em um projeto maior, essa etapa pode ser substituída por
    scikit-learn quando o ambiente estiver preparado para suas DLLs.
    """
    categories = sorted(data["categoria"].unique())
    category_matrix = pd.get_dummies(data["categoria"]).reindex(columns=categories, fill_value=0)
    features = np.column_stack([np.ones(len(data)), data["preco"].to_numpy(), category_matrix.to_numpy()])
    target = data["quantidade_vendida"].to_numpy(dtype=float)
    split = max(2, int(len(data) * 0.75))
    coefficients, *_ = np.linalg.lstsq(features[:split], target[:split], rcond=None)
    predictions = features[split:] @ coefficients
    actual = target[split:]
    mae = float(np.mean(np.abs(actual - predictions)))
    baseline = np.full_like(actual, target[:split].mean())
    baseline_mae = float(np.mean(np.abs(actual - baseline)))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 0.0 if denominator == 0 else float(1 - np.sum((actual - predictions) ** 2) / denominator)
    metrics = {
        "mae": round(mae, 2),
        "baseline_mae": round(baseline_mae, 2),
        "r2": round(r2, 2),
    }
    return coefficients, metrics


def predict_quantity(data: pd.DataFrame, preco: float, categoria: str) -> float:
    """Estima a quantidade vendida para um preço e categoria conhecidos."""
    coefficients, _ = train_model(data)
    categories = sorted(data["categoria"].unique())
    category_values = [1.0 if category == categoria else 0.0 for category in categories]
    features = np.array([1.0, float(preco), *category_values])
    prediction = float(features @ coefficients)
    return max(0.0, prediction)


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    _, metrics = train_model(data)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

