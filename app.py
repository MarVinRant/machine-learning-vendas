from pathlib import Path

import pandas as pd
import streamlit as st

from model import predict_quantity, train_model


BASE_DIR = Path(__file__).parent
data = pd.read_csv(BASE_DIR / "data" / "products.csv")
_, metrics = train_model(data)

st.set_page_config(page_title="Previsão de vendas", page_icon="🤖", layout="wide")
st.title("🤖 Previsão de vendas")
st.caption("Demonstração didática de regressão linear com NumPy e dados de produtos.")

metric_columns = st.columns(3)
metric_columns[0].metric("MAE do modelo", f"{metrics['mae']:.2f}")
metric_columns[1].metric("MAE do baseline", f"{metrics['baseline_mae']:.2f}")
metric_columns[2].metric("R²", f"{metrics['r2']:.2f}")

st.divider()
st.subheader("Faça uma estimativa")
with st.form("prediction_form"):
    category = st.selectbox("Categoria", sorted(data["categoria"].unique()))
    price = st.number_input("Preço (R$)", min_value=0.0, value=199.90, step=10.0)
    submitted = st.form_submit_button("Estimar quantidade")

if submitted:
    prediction = predict_quantity(data, price, category)
    st.success(f"Estimativa para {category}: **{prediction:.0f} itens vendidos**")

st.info(
    "Este é um projeto educacional. O conjunto de dados é pequeno e o resultado não deve "
    "ser usado para decisões reais sem mais dados, validação temporal e comparação com outros modelos."
)

with st.expander("Ver dados usados no treinamento"):
    st.dataframe(data, use_container_width=True, hide_index=True)
