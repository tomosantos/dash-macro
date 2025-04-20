#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para coleta de dados macroeconômicos do Brasil:
- IPCA (inflação)
- Taxa Selic
- PIB
- Câmbio (USD/BRL)

Fontes:
- Banco Central do Brasil (BCB)
- Instituto Brasileiro de Geografia e Estatística (IBGE)
- Yahoo Finance (para dados de câmbio)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from bcb import sgs
import warnings

# Ignorar avisos
warnings.filterwarnings('ignore')

# Criar diretório para armazenar os dados
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_bcb_data():
    """
    Coleta dados do Banco Central do Brasil usando a biblioteca python-bcb.
    
    Códigos das séries:
    - 433: IPCA - índice mensal
    - 4189: Taxa Selic - Meta definida pelo COPOM (% a.a.)
    - 4380: PIB mensal - Valores correntes (R$ milhões)
    - 1: Taxa de câmbio - Livre - Dólar americano (compra) - Média de período - mensal
    """
    print("Coletando dados do Banco Central do Brasil...")
    
    # Definir período de coleta (10 anos de dados históricos)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    
    # Códigos das séries temporais do BCB
    series_codes = {
        'ipca': 433,       # IPCA - índice mensal
        'selic': 4189,     # Taxa Selic - Meta definida pelo COPOM (% a.a.)
        'pib': 4380,       # PIB mensal - Valores correntes (R$ milhões)
        'cambio': 1        # Taxa de câmbio - Livre - Dólar americano (compra) - Média de período - mensal
    }
    
    # Coletar dados
    data = {}
    for name, code in series_codes.items():
        try:
            df = sgs.get(code, start=start_date, end=end_date)
            # Renomear a coluna para o nome do indicador
            df.columns = [name]
            data[name] = df
            print(f"Dados de {name} coletados com sucesso.")
            
            # Salvar dados em CSV
            df.to_csv(os.path.join(DATA_DIR, f'{name}.csv'))
            print(f"Dados de {name} salvos em {os.path.join(DATA_DIR, f'{name}.csv')}")
        except Exception as e:
            print(f"Erro ao coletar dados de {name}: {e}")
    
    return data

def get_yahoo_finance_data():
    """
    Coleta dados de câmbio (USD/BRL) do Yahoo Finance como fonte alternativa.
    """
    print("Coletando dados de câmbio do Yahoo Finance...")
    
    # Definir período de coleta (10 anos de dados históricos)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    
    try:
        # Baixar dados do par USD/BRL
        data = yf.download('USDBRL=X', start=start_date, end=end_date)
        
        # Selecionar apenas o preço de fechamento
        cambio_yf = data[['Close']].copy()
        cambio_yf.columns = ['cambio_yf']
        
        # Salvar dados em CSV
        cambio_yf.to_csv(os.path.join(DATA_DIR, 'cambio_yf.csv'))
        print(f"Dados de câmbio do Yahoo Finance salvos em {os.path.join(DATA_DIR, 'cambio_yf.csv')}")
        
        return cambio_yf
    except Exception as e:
        print(f"Erro ao coletar dados de câmbio do Yahoo Finance: {e}")
        return None

def plot_data(data):
    """
    Gera gráficos para visualização dos dados coletados.
    """
    print("Gerando gráficos para visualização dos dados...")
    
    # Configurar estilo dos gráficos
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Criar diretório para armazenar os gráficos
    PLOTS_DIR = os.path.join(DATA_DIR, 'plots')
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    
    # Plotar cada série temporal
    for name, df in data.items():
        plt.figure()
        plt.plot(df.index, df[name])
        plt.title(f'Série Histórica - {name.upper()}')
        plt.xlabel('Data')
        plt.ylabel(name.upper())
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'{name}_historico.png'))
        plt.close()
    
    print(f"Gráficos salvos no diretório {PLOTS_DIR}")

def analyze_data(data):
    """
    Realiza análise exploratória dos dados coletados.
    """
    print("Realizando análise exploratória dos dados...")
    
    # Criar diretório para armazenar os resultados da análise
    ANALYSIS_DIR = os.path.join(DATA_DIR, 'analysis')
    if not os.path.exists(ANALYSIS_DIR):
        os.makedirs(ANALYSIS_DIR)
    
    # Análise estatística básica
    for name, df in data.items():
        # Estatísticas descritivas
        stats = df.describe()
        stats.to_csv(os.path.join(ANALYSIS_DIR, f'{name}_stats.csv'))
        
        # Verificar valores ausentes
        missing = df.isnull().sum()
        
        # Calcular variação percentual
        if len(df) > 1:
            df[f'{name}_var_pct'] = df[name].pct_change() * 100
        
        # Salvar resultados da análise
        with open(os.path.join(ANALYSIS_DIR, f'{name}_analysis.txt'), 'w') as f:
            f.write(f"Análise de {name.upper()}\n")
            f.write("="*50 + "\n\n")
            f.write("Estatísticas Descritivas:\n")
            f.write(str(stats) + "\n\n")
            f.write("Valores Ausentes:\n")
            f.write(str(missing) + "\n\n")
            f.write(f"Período: {df.index.min()} a {df.index.max()}\n")
            f.write(f"Número de observações: {len(df)}\n")
    
    print(f"Análise exploratória concluída. Resultados salvos em {ANALYSIS_DIR}")

def main():
    """
    Função principal para coleta e processamento dos dados.
    """
    print("Iniciando coleta de dados macroeconômicos...")
    
    # Coletar dados do BCB
    bcb_data = get_bcb_data()
    
    # Coletar dados de câmbio do Yahoo Finance como fonte alternativa
    cambio_yf = get_yahoo_finance_data()
    
    # Se os dados do BCB foram coletados com sucesso, gerar gráficos e análises
    if bcb_data:
        # Plotar dados
        plot_data(bcb_data)
        
        # Analisar dados
        analyze_data(bcb_data)
    
    print("Coleta de dados concluída!")

if __name__ == "__main__":
    main()
