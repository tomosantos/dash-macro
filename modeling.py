#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para implementação de modelos preditivos para indicadores macroeconômicos:
- ARIMA
- Prophet
- Modelos de Machine Learning (Random Forest, XGBoost)

Os modelos são treinados com dados históricos e utilizados para gerar previsões
para os próximos 6-12 meses.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import pickle

# Modelos de séries temporais
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# Modelos de machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Ignorar avisos
warnings.filterwarnings('ignore')

# Diretórios
DATA_DIR = 'data'
MODELS_DIR = 'models'
FORECAST_DIR = 'forecasts'

# Criar diretórios se não existirem
for directory in [DATA_DIR, MODELS_DIR, FORECAST_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data():
    """
    Carrega os dados coletados anteriormente.
    """
    print("Carregando dados...")
    
    data = {}
    indicators = ['ipca', 'selic', 'pib', 'cambio']
    
    for indicator in indicators:
        file_path = os.path.join(DATA_DIR, f'{indicator}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            data[indicator] = df
            print(f"Dados de {indicator} carregados: {len(df)} registros")
        else:
            print(f"Arquivo {file_path} não encontrado")
    
    return data

def prepare_data_for_ml(df, indicator, window_size=6):
    """
    Prepara os dados para modelos de machine learning, criando features baseadas em lags.
    
    Args:
        df: DataFrame com os dados
        indicator: Nome do indicador
        window_size: Número de lags a serem usados como features
    
    Returns:
        X_train, y_train, X_test, y_test, scaler
    """
    # Criar cópia para não modificar o original
    data = df.copy()
    
    # Criar features baseadas em lags
    for i in range(1, window_size + 1):
        data[f'{indicator}_lag_{i}'] = data[indicator].shift(i)
    
    # Adicionar features de mês e ano
    data['month'] = data.index.month
    data['year'] = data.index.year
    
    # Remover linhas com NaN (devido aos lags)
    data = data.dropna()
    
    # Separar features e target
    X = data.drop(columns=[indicator])
    y = data[indicator]
    
    # Dividir em treino e teste (80% treino, 20% teste)
    train_size = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test, X_train.columns, scaler

def train_arima(df, indicator, order=(1,1,1), seasonal_order=(0,0,0,0)):
    """
    Treina um modelo ARIMA/SARIMA para o indicador especificado.
    
    Args:
        df: DataFrame com os dados
        indicator: Nome do indicador
        order: Ordem do modelo ARIMA (p,d,q)
        seasonal_order: Ordem sazonal (P,D,Q,s)
    
    Returns:
        Modelo treinado
    """
    print(f"Treinando modelo ARIMA para {indicator}...")
    
    # Preparar dados
    y = df[indicator]
    
    # Treinar modelo
    if sum(seasonal_order) > 0:
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order)
    else:
        model = ARIMA(y, order=order)
    
    fitted_model = model.fit()
    
    # Salvar modelo
    with open(os.path.join(MODELS_DIR, f'arima_{indicator}.pkl'), 'wb') as f:
        pickle.dump(fitted_model, f)
    
    print(f"Modelo ARIMA para {indicator} treinado e salvo")
    
    return fitted_model

def train_prophet(df, indicator):
    """
    Treina um modelo Prophet para o indicador especificado.
    
    Args:
        df: DataFrame com os dados
        indicator: Nome do indicador
    
    Returns:
        Modelo treinado
    """
    print(f"Treinando modelo Prophet para {indicator}...")
    
    # Preparar dados no formato exigido pelo Prophet (ds, y)
    prophet_df = df.reset_index()
    prophet_df.columns = ['ds', 'y']
    
    # Treinar modelo
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)
    
    # Salvar modelo
    with open(os.path.join(MODELS_DIR, f'prophet_{indicator}.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Modelo Prophet para {indicator} treinado e salvo")
    
    return model

def train_random_forest(X_train, y_train, indicator):
    """
    Treina um modelo Random Forest para o indicador especificado.
    
    Args:
        X_train: Features de treinamento
        y_train: Target de treinamento
        indicator: Nome do indicador
    
    Returns:
        Modelo treinado
    """
    print(f"Treinando modelo Random Forest para {indicator}...")
    
    # Treinar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Salvar modelo
    with open(os.path.join(MODELS_DIR, f'rf_{indicator}.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Modelo Random Forest para {indicator} treinado e salvo")
    
    return model

def train_xgboost(X_train, y_train, indicator):
    """
    Treina um modelo XGBoost para o indicador especificado.
    
    Args:
        X_train: Features de treinamento
        y_train: Target de treinamento
        indicator: Nome do indicador
    
    Returns:
        Modelo treinado
    """
    print(f"Treinando modelo XGBoost para {indicator}...")
    
    # Treinar modelo
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Salvar modelo
    with open(os.path.join(MODELS_DIR, f'xgb_{indicator}.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Modelo XGBoost para {indicator} treinado e salvo")
    
    return model

def evaluate_model(model, X_test, y_test, model_type, indicator):
    """
    Avalia o desempenho do modelo.
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste
        model_type: Tipo do modelo ('arima', 'prophet', 'rf', 'xgb')
        indicator: Nome do indicador
    
    Returns:
        Métricas de avaliação
    """
    print(f"Avaliando modelo {model_type} para {indicator}...")
    
    # Fazer previsões
    if model_type == 'arima':
        y_pred = model.forecast(len(y_test))
    elif model_type == 'prophet':
        future = model.make_future_dataframe(periods=len(y_test), freq='M')
        forecast = model.predict(future)
        y_pred = forecast.tail(len(y_test))['yhat'].values
    else:  # rf ou xgb
        y_pred = model.predict(X_test)
    
    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Salvar métricas
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    # Salvar métricas em arquivo
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(MODELS_DIR, f'{model_type}_{indicator}_metrics.csv'), index=False)
    
    print(f"Métricas para {model_type} - {indicator}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return metrics

def generate_forecast(models, indicator, forecast_horizon=12):
    """
    Gera previsões para o horizonte especificado usando todos os modelos treinados.
    
    Args:
        models: Dicionário com os modelos treinados
        indicator: Nome do indicador
        forecast_horizon: Horizonte de previsão em meses
    
    Returns:
        DataFrame com as previsões
    """
    print(f"Gerando previsões para {indicator} (horizonte: {forecast_horizon} meses)...")
    
    # Data atual
    last_date = models['data'][indicator].index[-1]
    
    # Datas futuras
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_horizon, freq='M')
    
    # DataFrame para armazenar as previsões
    forecasts = pd.DataFrame(index=future_dates)
    
    # Gerar previsões com ARIMA
    if 'arima' in models:
        arima_forecast = models['arima'].forecast(steps=forecast_horizon)
        forecasts['arima'] = arima_forecast.values
    
    # Gerar previsões com Prophet
    if 'prophet' in models:
        future = models['prophet'].make_future_dataframe(periods=forecast_horizon, freq='M')
        prophet_forecast = models['prophet'].predict(future)
        forecasts['prophet'] = prophet_forecast.tail(forecast_horizon)['yhat'].values
        forecasts['prophet_lower'] = prophet_forecast.tail(forecast_horizon)['yhat_lower'].values
        forecasts['prophet_upper'] = prophet_forecast.tail(forecast_horizon)['yhat_upper'].values
    
    # Inicializar colunas para RF e XGBoost
    forecasts['rf'] = np.nan
    forecasts['xgb'] = np.nan
    
    # Gerar previsões com Random Forest e XGBoost
    if 'rf' in models and 'feature_names' in models and 'scaler' in models:
        # Preparar dados para previsão
        last_values = models['data'][indicator].tail(models['window_size']).values
        
        for i in range(forecast_horizon):
            # Criar features para o próximo mês
            next_month = future_dates[i]
            features = {}
            
            # Lags
            for j in range(1, models['window_size'] + 1):
                if i - j + 1 < 0:  # Usar valores históricos
                    lag_idx = models['window_size'] - j + i
                    features[f'{indicator}_lag_{j}'] = last_values[lag_idx]
                else:  # Usar valores previstos
                    features[f'{indicator}_lag_{j}'] = forecasts['rf'].iloc[i-j]
            
            # Mês e ano
            features['month'] = next_month.month
            features['year'] = next_month.year
            
            # Criar DataFrame com as features
            X_next = pd.DataFrame([features])
            X_next = X_next[models['feature_names']]  # Garantir a ordem correta
            
            # Normalizar
            X_next_scaled = models['scaler'].transform(X_next)
            
            # Fazer previsões
            rf_pred = models['rf'].predict(X_next_scaled)[0]
            xgb_pred = models['xgb'].predict(X_next_scaled)[0]
            
            # Armazenar previsões
            forecasts.loc[next_month, 'rf'] = rf_pred
            forecasts.loc[next_month, 'xgb'] = xgb_pred
    
    # Calcular média das previsões (ensemble)
    model_columns = [col for col in forecasts.columns if col not in ['prophet_lower', 'prophet_upper']]
    forecasts['ensemble'] = forecasts[model_columns].mean(axis=1)
    
    # Salvar previsões
    forecasts.to_csv(os.path.join(FORECAST_DIR, f'{indicator}_forecast.csv'))
    
    # Plotar previsões
    plt.figure(figsize=(12, 8))
    
    # Dados históricos
    historical = models['data'][indicator]
    plt.plot(historical.index, historical.values, label='Histórico', color='black')
    
    # Previsões
    for col in model_columns + ['ensemble']:
        plt.plot(forecasts.index, forecasts[col], label=col)
    
    # Intervalo de confiança do Prophet
    if 'prophet_lower' in forecasts.columns and 'prophet_upper' in forecasts.columns:
        plt.fill_between(forecasts.index, forecasts['prophet_lower'], forecasts['prophet_upper'], 
                         color='lightblue', alpha=0.3, label='Intervalo de Confiança (Prophet)')
    
    plt.title(f'Previsão de {indicator.upper()} - Próximos {forecast_horizon} meses')
    plt.xlabel('Data')
    plt.ylabel(indicator.upper())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Salvar gráfico
    plt.savefig(os.path.join(FORECAST_DIR, f'{indicator}_forecast.png'))
    plt.close()
    
    print(f"Previsões para {indicator} geradas e salvas")
    
    return forecasts

def main():
    """
    Função principal para treinamento dos modelos e geração de previsões.
    """
    print("Iniciando treinamento dos modelos preditivos...")
    
    # Carregar dados
    data = load_data()
    
    # Parâmetros para os modelos ARIMA
    arima_params = {
        'ipca': {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)},
        'selic': {'order': (2, 1, 2), 'seasonal_order': (0, 0, 0, 0)},
        'pib': {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)},
        'cambio': {'order': (1, 1, 1), 'seasonal_order': (0, 0, 0, 0)}
    }
    
    # Horizonte de previsão
    forecast_horizon = 12  # 12 meses
    
    # Para cada indicador
    for indicator, df in data.items():
        print(f"\nProcessando indicador: {indicator}")
        
        # Armazenar modelos e dados para previsão
        models = {'data': data, 'window_size': 6}
        
        # Treinar modelo ARIMA
        models['arima'] = train_arima(df, indicator, 
                                     order=arima_params[indicator]['order'], 
                                     seasonal_order=arima_params[indicator]['seasonal_order'])
        
        # Treinar modelo Prophet
        models['prophet'] = train_prophet(df, indicator)
        
        # Preparar dados para modelos de ML
        X_train, y_train, X_test, y_test, feature_names, scaler = prepare_data_for_ml(df, indicator)
        models['feature_names'] = feature_names
        models['scaler'] = scaler
        
        # Treinar modelo Random Forest
        models['rf'] = train_random_forest(X_train, y_train, indicator)
        
        # Treinar modelo XGBoost
        models['xgb'] = train_xgboost(X_train, y_train, indicator)
        
        # Avaliar modelos
        evaluate_model(models['arima'], None, y_test, 'arima', indicator)
        evaluate_model(models['prophet'], None, y_test, 'prophet', indicator)
        evaluate_model(models['rf'], X_test, y_test, 'rf', indicator)
        evaluate_model(models['xgb'], X_test, y_test, 'xgb', indicator)
        
        # Gerar previsões
        generate_forecast(models, indicator, forecast_horizon)
    
    print("\nTreinamento dos modelos e geração de previsões concluídos!")

if __name__ == "__main__":
    main()
