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

    Args:
        path (Path): Caminho para o arquivo `log_amplitudes.json`.

    Returns:
        pd.DataFrame: Dados estruturados por passo/configuração.
    """
    with path.open("r", encoding="utf-8") as f:
        dados = json.load(f)

    registros = []
    for grupo in dados:
        for entrada in grupo:
            registros.append(entrada)

    df = pd.DataFrame(registros)
    df.sort_values(by=["passo"], inplace=True)
    return df


def plot_probabilidade_total(df: pd.DataFrame) -> None:
    """
    Plota a soma das probabilidades em cada passo da simulação.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados da simulação.
    """
    plt.figure()
    df.groupby("passo")["probabilidade"].sum().plot(marker='o', title="Soma das Probabilidades por Passo")
    plt.xlabel("Passo")
    plt.ylabel("Probabilidade Total")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_caminho_dominante(df: pd.DataFrame) -> None:
    """
    Plota a configuração mais provável (dominante) ao longo dos passos.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados da simulação.
    """
    plt.figure()
    mais_provaveis = df.sort_values("probabilidade", ascending=False).groupby("passo").first().reset_index()
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

    Args:
        df (pd.DataFrame): DataFrame contendo os dados da simulação.
    """
    df_qf = df[df["estado"] == "qf"]
    plt.figure()
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

    Args:
        df (pd.DataFrame): DataFrame contendo os dados da simulação.
    """
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
    """
    caminho_log = Path("log_amplitudes.json")
    df = carregar_log(caminho_log)

    plot_probabilidade_total(df)
    plot_caminho_dominante(df)
    plot_estado_final_qf(df)
    plot_mapa_calor(df)


if __name__ == "__main__":
    main()
