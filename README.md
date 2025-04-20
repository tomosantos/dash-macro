# Dashboard de Previsões Macroeconômicas do Brasil

Este projeto implementa um dashboard interativo para visualização e previsão dos principais indicadores macroeconômicos do Brasil: IPCA (inflação), Taxa Selic, PIB e câmbio (USD/BRL).

## Funcionalidades

- **Coleta Automatizada de Dados**: Obtém dados históricos de fontes oficiais como Banco Central do Brasil (BACEN) e IBGE
- **Modelagem Preditiva**: Implementa múltiplos modelos (ARIMA, Prophet, Random Forest, XGBoost) para gerar previsões
- **Visualizações Interativas**: Gráficos dinâmicos com séries históricas e previsões futuras
- **Intervalos de Confiança**: Representação visual da incerteza nas previsões
- **Seleção de Indicadores**: Interface para escolher o indicador de interesse
- **Horizonte Ajustável**: Opção para definir o período de previsão desejado
- **Exportação de Dados**: Funcionalidade para exportar dados em CSV ou Excel
- **Geração de Relatórios**: Criação de relatórios detalhados em HTML
- **Atualização Automática**: Sistema de atualização semanal dos dados e retreinamento dos modelos

## Estrutura do Projeto

- `collect_data.py`: Script para coleta de dados históricos das fontes oficiais
- `modeling.py`: Implementação dos modelos preditivos e geração de previsões
- `app.py`: Aplicação Streamlit para o dashboard interativo
- `update_data.py`: Script para atualização automática semanal dos dados
- `crontab_setup.sh`: Configuração do cron job para execução automática
- `data/`: Diretório com dados históricos coletados
- `models/`: Diretório com modelos treinados e métricas de avaliação
- `forecasts/`: Diretório com previsões geradas
- `assets/`: Diretório com arquivos exportados e relatórios

## Tecnologias Utilizadas

- **Python**: Linguagem principal de programação
- **Pandas/NumPy**: Manipulação e análise de dados
- **Matplotlib/Seaborn/Plotly**: Visualização de dados
- **Statsmodels**: Implementação de modelos ARIMA
- **Prophet**: Biblioteca do Facebook para previsão de séries temporais
- **Scikit-learn**: Implementação de modelos de machine learning
- **XGBoost**: Algoritmo de gradient boosting para regressão
- **Streamlit**: Framework para criação do dashboard interativo

## Instalação e Execução

1. Clone o repositório
2. Crie um ambiente virtual Python:
   ```
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```
3. Instale as dependências:
   ```
   pip install pandas numpy matplotlib seaborn plotly scikit-learn statsmodels prophet streamlit yfinance python-bcb xgboost
   ```
4. Execute a coleta de dados:
   ```
   python collect_data.py
   ```
5. Treine os modelos:
   ```
   python modeling.py
   ```
6. Inicie o dashboard:
   ```
   streamlit run app.py
   ```

## Acesso ao Dashboard

O dashboard está disponível publicamente em: https://dash-macro-br.streamlit.app/


## Limitações e Considerações

- As previsões são baseadas em dados históricos e podem não capturar eventos extraordinários ou mudanças estruturais na economia
- Recomenda-se utilizar as previsões como referência e complementá-las com análises qualitativas
- A precisão dos modelos varia conforme o indicador e o horizonte de previsão

## Próximos Passos

- Implementação de modelos mais avançados (redes neurais, modelos híbridos)
- Inclusão de variáveis exógenas para melhorar a precisão das previsões
- Expansão para outros indicadores macroeconômicos
- Desenvolvimento de alertas automáticos para mudanças significativas nas previsões
