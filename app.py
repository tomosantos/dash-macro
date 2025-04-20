#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dashboard interativo para visualização de previsões macroeconômicas do Brasil:
- IPCA (inflação)
- Taxa Selic
- PIB
- Câmbio (USD/BRL)

O dashboard permite visualizar dados históricos e previsões futuras
para os próximos 6-12 meses, com intervalos de confiança.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pickle
from PIL import Image
import warnings

# Ignorar avisos
warnings.filterwarnings('ignore')

# Diretórios
DATA_DIR = 'data'
MODELS_DIR = 'models'
FORECAST_DIR = 'forecasts'
ASSETS_DIR = 'assets'

# Criar diretório de assets se não existir
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)

# Configurações da página
st.set_page_config(
    page_title="Dashboard de Previsões Macroeconômicas do Brasil",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para carregar dados
@st.cache_data(ttl=3600*24)  # Cache por 24 horas
def load_data():
    """
    Carrega os dados históricos e previsões.
    """
    data = {}
    indicators = ['ipca', 'selic', 'pib', 'cambio']
    
    for indicator in indicators:
        # Carregar dados históricos
        hist_file = os.path.join(DATA_DIR, f'{indicator}.csv')
        if os.path.exists(hist_file):
            df = pd.read_csv(hist_file, index_col=0, parse_dates=True)
            
            # Carregar previsões
            forecast_file = os.path.join(FORECAST_DIR, f'{indicator}_forecast.csv')
            if os.path.exists(forecast_file):
                forecast_df = pd.read_csv(forecast_file, index_col=0, parse_dates=True)
                
                # Combinar dados históricos e previsões
                data[indicator] = {
                    'historical': df,
                    'forecast': forecast_df
                }
            else:
                data[indicator] = {
                    'historical': df,
                    'forecast': None
                }
        else:
            st.error(f"Arquivo de dados históricos para {indicator} não encontrado.")
    
    return data

# Função para carregar métricas dos modelos
@st.cache_data(ttl=3600*24)  # Cache por 24 horas
def load_model_metrics():
    """
    Carrega as métricas de avaliação dos modelos.
    """
    metrics = {}
    indicators = ['ipca', 'selic', 'pib', 'cambio']
    model_types = ['arima', 'prophet', 'rf', 'xgb']
    
    for indicator in indicators:
        metrics[indicator] = {}
        for model_type in model_types:
            metrics_file = os.path.join(MODELS_DIR, f'{model_type}_{indicator}_metrics.csv')
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                metrics[indicator][model_type] = df
    
    return metrics

# Função para criar gráfico de dados históricos e previsões
def plot_forecast(data, indicator, models=None, confidence_interval=True):
    """
    Cria um gráfico interativo com dados históricos e previsões.
    
    Args:
        data: Dicionário com dados históricos e previsões
        indicator: Nome do indicador
        models: Lista de modelos para exibir (None para todos)
        confidence_interval: Se deve exibir intervalo de confiança
    """
    if models is None:
        models = ['arima', 'prophet', 'rf', 'xgb', 'ensemble']
    
    # Preparar dados
    historical = data[indicator]['historical']
    forecast = data[indicator]['forecast']
    
    # Criar figura
    fig = go.Figure()
    
    # Adicionar dados históricos
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical[indicator],
        mode='lines',
        name='Histórico',
        line=dict(color='black', width=2)
    ))
    
    # Adicionar previsões
    colors = {
        'arima': 'blue',
        'prophet': 'red',
        'rf': 'green',
        'xgb': 'purple',
        'ensemble': 'orange'
    }
    
    model_names = {
        'arima': 'ARIMA',
        'prophet': 'Prophet',
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'ensemble': 'Ensemble (média)'
    }
    
    for model in models:
        if model in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast[model],
                mode='lines',
                name=model_names.get(model, model),
                line=dict(color=colors.get(model, 'gray'), width=2, dash='dash')
            ))
    
    # Adicionar intervalo de confiança
    if confidence_interval and 'prophet_lower' in forecast.columns and 'prophet_upper' in forecast.columns:
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast['prophet_upper'],
            mode='lines',
            name='Limite Superior',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast['prophet_lower'],
            mode='lines',
            name='Intervalo de Confiança',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.3)'
        ))
    
    # Configurar layout
    title_map = {
        'ipca': 'IPCA (Inflação)',
        'selic': 'Taxa Selic',
        'pib': 'PIB (R$ milhões)',
        'cambio': 'Câmbio (USD/BRL)'
    }
    
    fig.update_layout(
        title=f"{title_map.get(indicator, indicator.upper())} - Histórico e Previsão",
        xaxis_title="Data",
        yaxis_title=title_map.get(indicator, indicator.upper()),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )
    
    return fig

# Função para criar tabela de métricas
def create_metrics_table(metrics, indicator):
    """
    Cria uma tabela com as métricas de avaliação dos modelos.
    
    Args:
        metrics: Dicionário com métricas dos modelos
        indicator: Nome do indicador
    """
    if indicator not in metrics:
        return None
    
    # Criar DataFrame para a tabela
    table_data = []
    model_names = {
        'arima': 'ARIMA',
        'prophet': 'Prophet',
        'rf': 'Random Forest',
        'xgb': 'XGBoost'
    }
    
    for model_type, df in metrics[indicator].items():
        if not df.empty:
            row = {
                'Modelo': model_names.get(model_type, model_type),
                'RMSE': df['rmse'].values[0],
                'MAE': df['mae'].values[0],
                'R²': df['r2'].values[0]
            }
            table_data.append(row)
    
    if table_data:
        return pd.DataFrame(table_data)
    else:
        return None

# Função para exportar dados
def export_data(data, indicator, format='csv'):
    """
    Exporta dados históricos e previsões para CSV ou Excel.
    
    Args:
        data: Dicionário com dados históricos e previsões
        indicator: Nome do indicador
        format: Formato de exportação ('csv' ou 'excel')
    
    Returns:
        Caminho do arquivo exportado
    """
    # Combinar dados históricos e previsões
    historical = data[indicator]['historical'].copy()
    forecast = data[indicator]['forecast'].copy()
    
    # Renomear colunas do histórico
    historical.columns = [f'{col}_historico' for col in historical.columns]
    
    # Criar DataFrame combinado
    combined = pd.concat([historical, forecast], axis=1)
    
    # Exportar
    export_dir = os.path.join(ASSETS_DIR, 'exports')
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format == 'csv':
        filename = os.path.join(export_dir, f'{indicator}_dados_e_previsoes_{timestamp}.csv')
        combined.to_csv(filename)
    else:  # excel
        filename = os.path.join(export_dir, f'{indicator}_dados_e_previsoes_{timestamp}.xlsx')
        combined.to_excel(filename)
    
    return filename

# Função para gerar relatório
def generate_report(data, indicator, selected_models):
    """
    Gera um relatório em HTML com análise dos dados e previsões.
    
    Args:
        data: Dicionário com dados históricos e previsões
        indicator: Nome do indicador
        selected_models: Lista de modelos selecionados
    
    Returns:
        Caminho do arquivo de relatório
    """
    # Preparar dados
    historical = data[indicator]['historical']
    forecast = data[indicator]['forecast']
    
    # Criar diretório para relatórios
    report_dir = os.path.join(ASSETS_DIR, 'reports')
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # Nome do arquivo
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(report_dir, f'{indicator}_relatorio_{timestamp}.html')
    
    # Mapear nomes dos indicadores
    indicator_names = {
        'ipca': 'IPCA (Inflação)',
        'selic': 'Taxa Selic',
        'pib': 'PIB',
        'cambio': 'Câmbio (USD/BRL)'
    }
    
    # Mapear nomes dos modelos
    model_names = {
        'arima': 'ARIMA',
        'prophet': 'Prophet',
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'ensemble': 'Ensemble (média)'
    }
    
    # Criar conteúdo HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Relatório de Previsões - {indicator_names.get(indicator, indicator.upper())}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .container {{ margin-bottom: 30px; }}
            .footer {{ margin-top: 50px; font-size: 0.8em; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <h1>Relatório de Previsões Macroeconômicas</h1>
        <h2>{indicator_names.get(indicator, indicator.upper())}</h2>
        <p>Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        
        <div class="container">
            <h3>Resumo dos Dados Históricos</h3>
            <table>
                <tr>
                    <th>Métrica</th>
                    <th>Valor</th>
                </tr>
                <tr>
                    <td>Período</td>
                    <td>{historical.index.min().strftime('%d/%m/%Y')} a {historical.index.max().strftime('%d/%m/%Y')}</td>
                </tr>
                <tr>
                    <td>Número de observações</td>
                    <td>{len(historical)}</td>
                </tr>
                <tr>
                    <td>Valor mínimo</td>
                    <td>{historical[indicator].min():.4f}</td>
                </tr>
                <tr>
                    <td>Valor máximo</td>
                    <td>{historical[indicator].max():.4f}</td>
                </tr>
                <tr>
                    <td>Média</td>
                    <td>{historical[indicator].mean():.4f}</td>
                </tr>
                <tr>
                    <td>Mediana</td>
                    <td>{historical[indicator].median():.4f}</td>
                </tr>
                <tr>
                    <td>Desvio padrão</td>
                    <td>{historical[indicator].std():.4f}</td>
                </tr>
            </table>
        </div>
        
        <div class="container">
            <h3>Previsões</h3>
            <p>Horizonte de previsão: {len(forecast)} meses</p>
            <p>Período: {forecast.index.min().strftime('%d/%m/%Y')} a {forecast.index.max().strftime('%d/%m/%Y')}</p>
            
            <h4>Modelos utilizados:</h4>
            <ul>
    """
    
    # Adicionar modelos selecionados
    for model in selected_models:
        if model in forecast.columns:
            html_content += f"<li>{model_names.get(model, model)}</li>\n"
    
    html_content += """
            </ul>
            
            <h4>Tabela de previsões:</h4>
            <table>
                <tr>
                    <th>Data</th>
    """
    
    # Adicionar cabeçalhos para cada modelo
    for model in selected_models:
        if model in forecast.columns:
            html_content += f"<th>{model_names.get(model, model)}</th>\n"
    
    html_content += """
                </tr>
    """
    
    # Adicionar linhas da tabela
    for date, row in forecast.iterrows():
        html_content += f"""
                <tr>
                    <td>{date.strftime('%d/%m/%Y')}</td>
        """
        
        for model in selected_models:
            if model in forecast.columns:
                html_content += f"<td>{row[model]:.4f}</td>\n"
        
        html_content += """
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="container">
            <h3>Análise e Conclusões</h3>
            <p>
                Este relatório apresenta as previsões para o indicador macroeconômico selecionado,
                utilizando diferentes modelos estatísticos e de machine learning. As previsões são
                baseadas em dados históricos e podem ser utilizadas como referência para tomada de
                decisões, mas estão sujeitas a incertezas e fatores externos não capturados pelos modelos.
            </p>
            <p>
                Recomenda-se acompanhar regularmente as atualizações dos dados e revisões das previsões,
                bem como considerar outros fatores econômicos, políticos e sociais que possam impactar
                o comportamento futuro do indicador.
            </p>
        </div>
        
        <div class="footer">
            <p>Dashboard de Previsões Macroeconômicas do Brasil</p>
            <p>Gerado automaticamente em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """
    
    # Salvar arquivo HTML
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return filename

# Interface principal do dashboard
def main():
    # Título e descrição
    st.title("Dashboard de Previsões Macroeconômicas do Brasil")
    st.markdown("""
    Este dashboard apresenta dados históricos e previsões para os principais indicadores 
    macroeconômicos do Brasil: IPCA (inflação), Taxa Selic, PIB e câmbio (USD/BRL).
    
    As previsões são geradas utilizando diferentes modelos estatísticos e de machine learning,
    com horizonte de 12 meses.
    """)
    
    # Carregar dados
    with st.spinner("Carregando dados..."):
        data = load_data()
        metrics = load_model_metrics()
    
    # Barra lateral
    st.sidebar.title("Configurações")
    
    # Seleção de indicador
    indicator = st.sidebar.selectbox(
        "Selecione o indicador:",
        options=['ipca', 'selic', 'pib', 'cambio'],
        format_func=lambda x: {
            'ipca': 'IPCA (Inflação)',
            'selic': 'Taxa Selic',
            'pib': 'PIB',
            'cambio': 'Câmbio (USD/BRL)'
        }.get(x, x.upper())
    )
    
    # Seleção de modelos
    available_models = ['arima', 'prophet', 'rf', 'xgb', 'ensemble']
    selected_models = st.sidebar.multiselect(
        "Selecione os modelos:",
        options=available_models,
        default=['ensemble'],
        format_func=lambda x: {
            'arima': 'ARIMA',
            'prophet': 'Prophet',
            'rf': 'Random Forest',
            'xgb': 'XGBoost',
            'ensemble': 'Ensemble (média)'
        }.get(x, x)
    )
    
    # Opção de intervalo de confiança
    show_ci = st.sidebar.checkbox("Mostrar intervalo de confiança", value=True)
    
    # Horizonte de previsão
    if data[indicator]['forecast'] is not None:
        max_horizon = len(data[indicator]['forecast'])
        horizon = st.sidebar.slider(
            "Horizonte de previsão (meses):",
            min_value=1,
            max_value=max_horizon,
            value=max_horizon
        )
    else:
        horizon = 12
    
    # Exportação de dados
    st.sidebar.subheader("Exportar Dados")
    export_format = st.sidebar.radio("Formato:", options=['CSV', 'Excel'])
    
    if st.sidebar.button("Exportar Dados"):
        with st.spinner("Exportando dados..."):
            filename = export_data(data, indicator, format=export_format.lower())
            st.sidebar.success(f"Dados exportados para {filename}")
    
    # Geração de relatório
    st.sidebar.subheader("Gerar Relatório")
    
    if st.sidebar.button("Gerar Relatório"):
        with st.spinner("Gerando relatório..."):
            filename = generate_report(data, indicator, selected_models)
            st.sidebar.success(f"Relatório gerado: {filename}")
    
    # Conteúdo principal
    if indicator in data and data[indicator]['historical'] is not None:
        # Dividir em abas
        tab1, tab2, tab3 = st.tabs(["Visualização", "Métricas", "Dados"])
        
        with tab1:
            # Gráfico principal
            if data[indicator]['forecast'] is not None:
                # Limitar horizonte de previsão
                forecast_limited = data[indicator]['forecast'].iloc[:horizon].copy()
                data_for_plot = {
                    indicator: {
                        'historical': data[indicator]['historical'],
                        'forecast': forecast_limited
                    }
                }
                
                fig = plot_forecast(data_for_plot, indicator, models=selected_models, confidence_interval=show_ci)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Dados de previsão não disponíveis para este indicador.")
        
        with tab2:
            # Tabela de métricas
            st.subheader("Métricas de Avaliação dos Modelos")
            metrics_table = create_metrics_table(metrics, indicator)
            
            if metrics_table is not None:
                st.dataframe(metrics_table, use_container_width=True)
                
                # Explicação das métricas
                with st.expander("Explicação das métricas"):
                    st.markdown("""
                    - **RMSE (Root Mean Square Error)**: Raiz do erro quadrático médio. Quanto menor, melhor.
                    - **MAE (Mean Absolute Error)**: Erro absoluto médio. Quanto menor, melhor.
                    - **R² (Coeficiente de determinação)**: Indica quanto da variância dos dados é explicada pelo modelo. Quanto mais próximo de 1, melhor.
                    """)
            else:
                st.info("Métricas não disponíveis para este indicador.")
        
        with tab3:
            # Dados históricos e previsões
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dados Históricos")
                st.dataframe(data[indicator]['historical'], use_container_width=True)
            
            with col2:
                st.subheader("Previsões")
                if data[indicator]['forecast'] is not None:
                    st.dataframe(data[indicator]['forecast'], use_container_width=True)
                else:
                    st.info("Previsões não disponíveis.")
    else:
        st.error("Dados não disponíveis para o indicador selecionado.")
    
    # Rodapé
    st.markdown("---")
    st.markdown("""
    **Dashboard de Previsões Macroeconômicas do Brasil**
    
    Desenvolvido com Streamlit, Python e modelos de séries temporais.
    
    Dados atualizados semanalmente a partir de fontes oficiais (BACEN e IBGE).
    """)

if __name__ == "__main__":
    main()
