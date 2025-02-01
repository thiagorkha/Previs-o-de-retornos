import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

def analyze_stock_with_prophet(stock_ticker, forecast_horizon=15, test_size=0.2):
    """
    Analisa uma ação com Prophet, incluindo validação cruzada e métricas de desempenho.
    
    Parâmetros:
    - stock_ticker: Ticker da ação (ex: 'PETR4.SA' para Petrobras).
    - forecast_horizon: Número de dias úteis para previsão (padrão: 15).
    - test_size: Proporção dos dados para teste (padrão: 20%).
    """
    
    # Baixar dados (últimos 2 anos)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    try:
        data = yf.download(stock_ticker, start=start_date, end=end_date)['Adj Close']
    except Exception as e:
        st.error(f"Erro ao baixar dados para {stock_ticker}: {e}")
        return
    
    # Verificar se há dados suficientes
    if len(data) < 30:
        st.error(f"Erro: Não há dados suficientes para {stock_ticker}.")
        return
    
    # Calcular retornos diários (1-dimensional)
    returns = data.pct_change().dropna().reset_index()
    returns.columns = ['ds', 'y']  # Renomear para o formato do Prophet
    
    # Verificar se há retornos válidos
    if len(returns) < 30:
        st.error(f"Erro: Não há retornos suficientes para {stock_ticker}.")
        return
    
    # Dividir em treino e teste
    train_size = int(len(returns) * (1 - test_size))
    train = returns.iloc[:train_size]
    test = returns.iloc[train_size:]
    
    # Treinar modelo Prophet
    model = Prophet(interval_width=0.95)  # Intervalo de confiança de 95%
    model.fit(train[['ds', 'y']])  # Garantir que as colunas são 1-dimensional
    
    # Validação cruzada
    try:
        df_cv = cross_validation(model, initial=f'{train_size} days', period='30 days', horizon=f'{forecast_horizon} days')
        df_p = performance_metrics(df_cv)
        rmse = np.mean(df_p['rmse'])
        mae = np.mean(df_p['mae'])
        st.write(f"Médias - CV RMSE: {rmse:.4f}, CV MAE: {mae:.4f}")
    except Exception as e:
        st.error(f"Erro na validação cruzada: {e}")
        return
    
    # Prever para o horizonte especificado
    future = model.make_future_dataframe(periods=forecast_horizon, freq='B')  # Dias úteis
    forecast = model.predict(future)
    
    # Filtrar previsões
    forecast_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon)
    
    # Calcular retorno acumulado
    cumulative_return = np.prod(1 + forecast_values['yhat']) - 1
    lower_bound = np.prod(1 + forecast_values['yhat_lower']) - 1
    upper_bound = np.prod(1 + forecast_values['yhat_upper']) - 1
    
    # Resultados
    st.write(f"[Análise para {stock_ticker} - {datetime.today().strftime('%Y-%m-%d')}]")
    st.write(f"- Retorno Esperado ({forecast_horizon} dias): {cumulative_return*100:.2f}%")
    st.write(f"- Intervalo de Confiança: [{lower_bound*100:.2f}%, {upper_bound*100:.2f}%]")
    
    # Visualização
    fig = model.plot(forecast, uncertainty=True)
    plt.title(f'Previsões para {stock_ticker} ({forecast_horizon} dias)')
    plt.xlabel('Data')
    plt.ylabel('Retorno Diário')
    st.pyplot(fig)

# Interface do Streamlit
st.title("Análise de Ações com Prophet")

# Inputs do usuário
stock_ticker = st.text_input("Digite o ticker da ação (ex: PETR4.SA):", "PETR4.SA")
forecast_horizon = st.number_input("Número de dias úteis para previsão:", min_value=1, value=15)
test_size = st.slider("Proporção dos dados para teste:", min_value=0.1, max_value=0.5, value=0.2)

# Botão para rodar a análise
if st.button("Analisar"):
    analyze_stock_with_prophet(stock_ticker, forecast_horizon, test_size)
