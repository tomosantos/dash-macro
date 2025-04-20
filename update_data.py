#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para atualização automática dos dados macroeconômicos e retreinamento dos modelos.
Este script é projetado para ser executado semanalmente via cron job.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import subprocess
import traceback

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('update_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('update_data')

def run_script(script_path):
    """
    Executa um script Python e registra a saída.
    
    Args:
        script_path: Caminho para o script a ser executado
    
    Returns:
        bool: True se o script foi executado com sucesso, False caso contrário
    """
    logger.info(f"Executando script: {script_path}")
    
    try:
        result = subprocess.run(
            ['python', script_path],
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(f"Saída do script: {result.stdout}")
        
        if result.stderr:
            logger.warning(f"Erros/avisos do script: {result.stderr}")
        
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao executar o script {script_path}: {e}")
        logger.error(f"Saída de erro: {e.stderr}")
        return False
    
    except Exception as e:
        logger.error(f"Exceção ao executar o script {script_path}: {e}")
        logger.error(traceback.format_exc())
        return False

def update_data():
    """
    Função principal para atualização dos dados e modelos.
    """
    logger.info("Iniciando atualização semanal dos dados macroeconômicos")
    
    # Registrar data e hora da atualização
    update_time = datetime.now()
    logger.info(f"Data e hora da atualização: {update_time}")
    
    # Passo 1: Coletar novos dados
    logger.info("Passo 1: Coletando novos dados macroeconômicos")
    collect_success = run_script('collect_data.py')
    
    if not collect_success:
        logger.error("Falha na coleta de dados. Abortando atualização.")
        return False
    
    # Passo 2: Retreinar modelos com os novos dados
    logger.info("Passo 2: Retreinando modelos preditivos")
    modeling_success = run_script('modeling.py')
    
    if not modeling_success:
        logger.error("Falha no retreinamento dos modelos. Atualização incompleta.")
        return False
    
    # Passo 3: Registrar sucesso da atualização
    logger.info("Atualização semanal concluída com sucesso!")
    
    # Criar arquivo de status da atualização
    status = {
        'last_update': update_time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'success',
        'next_update': (update_time + timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    }
    
    status_df = pd.DataFrame([status])
    status_df.to_csv('update_status.csv', index=False)
    
    logger.info(f"Próxima atualização programada para: {status['next_update']}")
    
    return True

if __name__ == "__main__":
    update_data()
