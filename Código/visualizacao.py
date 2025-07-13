"""
visualizacao.py

Este módulo realiza a visualização dos resultados de uma simulação
de Máquina de Turing Quântica (MTQ), a partir do arquivo JSON gerado
(`log_amplitudes.json`). Os gráficos incluem:

1. Soma das probabilidades por passo
2. Caminho mais provável do sistema
3. Evolução da probabilidade do estado final ('qf')
4. Mapa de calor da probabilidade por configuração

Autor: Emanuel Lopes
Data: Julho de 2025
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def carregar_log(path: Path) -> pd.DataFrame:
    """
    Carrega o log da MTQ em formato JSON e converte para DataFrame plano.

    O arquivo JSON é esperado no formato de uma lista de listas, onde cada sublista
    contém os estados da MTQ em um determinado passo. Esta função "achata" essa
    estrutura para um DataFrame tabular, facilitando a análise e visualização.

    Args:
        path (Path): Caminho para o arquivo `log_amplitudes.json`.

    Returns:
        pd.DataFrame: Dados estruturados por passo/configuração, contendo
                      colunas como 'passo', 'estado', 'fita', 'probabilidade', etc.
    """
    with path.open("r", encoding="utf-8") as f:
        dados = json.load(f)

    registros = []
    # Itera sobre cada grupo de entradas (cada "passo" de simulação)
    for grupo in dados:
        # Adiciona cada entrada individual (Configuração da MTQ) à lista de registros
        for entrada in grupo:
            registros.append(entrada)

    # Converte a lista de registros em um DataFrame do pandas
    df = pd.DataFrame(registros)
    # Garante que o DataFrame esteja ordenado pelos passos da simulação
    df.sort_values(by=["passo"], inplace=True)
    return df


def plot_probabilidade_total(df: pd.DataFrame) -> None:
    """
    Plota a soma das probabilidades em cada passo da simulação.

    Este gráfico serve para verificar a conservação da norma quântica (a soma
    das probabilidades de todos os estados em superposição deve ser sempre 1).
    Desvios significativos podem indicar problemas na simulação.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados da simulação.
    """
    plt.figure()
    # Agrupa os dados pelo 'passo' e soma as 'probabilidades' para cada passo
    df.groupby("passo")["probabilidade"].sum().plot(marker='o', title="Soma das Probabilidades por Passo")
    plt.xlabel("Passo")
    plt.ylabel("Probabilidade Total")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_caminho_dominante(df: pd.DataFrame) -> None:
    """
    Plota a configuração mais provável (dominante) ao longo dos passos.

    Este gráfico mostra como a probabilidade se concentra na "trajetória"
    principal da MTQ ao longo do tempo, evidenciando o efeito da interferência
    construtiva e a eventual convergência do sistema.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados da simulação.
    """
    plt.figure()
    # Encontra a configuração com a maior probabilidade em cada passo
    mais_provaveis = df.sort_values("probabilidade", ascending=False).groupby("passo").first().reset_index()
    # Plota a probabilidade da configuração dominante em função do passo
    plt.plot(mais_provaveis["passo"], mais_provaveis["probabilidade"],
             marker='o', color="purple", label="Configuração dominante")
    plt.title("Caminho Mais Provável ao Longo dos Passos")
    plt.xlabel("Passo")
    plt.ylabel("Probabilidade da Configuração Dominante")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_estado_final_qf(df: pd.DataFrame) -> None:
    """
    Plota a evolução da probabilidade do estado de aceitação 'qf'.

    Este gráfico foca especificamente no estado final de aceitação ('qf') da
    Máquina de Turing Quântica, mostrando como sua probabilidade evolui
    (idealmente, se aproxima de 1) ao longo da simulação, indicando a aceitação
    bem-sucedida da entrada.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados da simulação.
    """
    # Filtra o DataFrame para incluir apenas as entradas onde o estado é 'qf'
    df_qf = df[df["estado"] == "qf"]
    plt.figure()
    # Plota a probabilidade do estado 'qf' em função do passo
    plt.plot(df_qf["passo"], df_qf["probabilidade"], marker='o', color='green')
    plt.title("Probabilidade do Estado Final (qf)")
    plt.xlabel("Passo")
    plt.ylabel("Probabilidade")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_mapa_calor(df: pd.DataFrame) -> None:
    """
    Plota um mapa de calor das probabilidades por (estado, fita) ao longo dos passos.

    O mapa de calor oferece uma visão abrangente de como a probabilidade se distribui
    entre todas as configurações (estado interno e conteúdo da fita) da MTQ em
    cada passo. Isso é útil para observar a "difusão" e "concentração" das
    probabilidades devido à superposição e interferência.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados da simulação.
    """
    # Pivota o DataFrame para ter 'config_id' como índice, 'passo' como colunas
    # e 'probabilidade' como valores. Preenche valores ausentes com 0.
    pivot = df.pivot_table(index=["estado", "fita"], columns="passo",
                           values="probabilidade", fill_value=0)
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="viridis", annot=False, cbar_kws={'label': 'Probabilidade'})
    plt.title("Mapa de Calor: Probabilidade por Configuração e Passo")
    plt.xlabel("Passo")
    plt.ylabel("Configuração (Estado - Fita)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Função principal. Executa todos os gráficos com base no log de amplitudes.

    Esta função coordena o carregamento dos dados de simulação e a chamada
    de cada função de plotagem para gerar todas as visualizações definidas
    neste módulo.
    """
    # Define o caminho para o arquivo de log gerado pela simulação da MTQ
    caminho_log = Path("log_amplitudes.json")

    if not caminho_log.exists():
        print(f"Erro: O arquivo de log '{caminho_log}' não foi encontrado.")
        print("Certifique-se de executar 'main.py' primeiro para gerar o log.")
        return

    # Carrega os dados do log em um DataFrame
    df = carregar_log(caminho_log)

    # Gera cada tipo de gráfico
    plot_probabilidade_total(df)
    plot_caminho_dominante(df)
    plot_estado_final_qf(df)
    plot_mapa_calor(df)


if __name__ == "__main__":
    main()
