# quantum_extensions.py
"""
quantum_extensions.py

Este módulo adiciona extensões quânticas à Máquina de Turing Quântica (MTQ),
incluindo suporte a vetores de estado com amplitudes complexas, fitas em superposição,
operadores unitários simulados, oráculo de Grover, difusão e decoerência (ruído quântico).

Autor: Emanuel Lopes Silva
Data: Julho de 2025
"""

import numpy as np
import random
import cmath
from collections import defaultdict

# -----------------------------------------------------------------------------
# 1. QuantumRegister: representa o vetor de estado com amplitudes complexas
# -----------------------------------------------------------------------------
class QuantumRegister:
    """
    Representa o vetor de estado quântico como um dicionário esparso.

    O estado da máquina é representado por um mapeamento de configurações
    da Máquina de Turing Quântica (MTQ) para amplitudes complexas.
    Uma configuração é uma tupla (estado_interno, posição_cabeça, conteúdo_fita).

    { (estado, cabeça, fita): amplitude complexa }

    Métodos:
        - set: define a amplitude manualmente para uma dada configuração.
        - normalize: ajusta as amplitudes para que a soma dos quadrados de seus
                     módulos seja 1 (condição de normalização em mecânica quântica).
        - measure: colapsa o vetor de estado para uma única configuração,
                   escolhida probabilisticamente de acordo com a regra de Born.
        - apply_unitary: aplica um operador unitário (representado como dicionário)
                         ao vetor de estado, simulando a evolução temporal do sistema.

    Atributos:
        states (defaultdict): Um dicionário onde as chaves são as configurações da MTQ
                              e os valores são suas amplitudes complexas. Usa defaultdict
                              para inicializar amplitudes como 0j se a configuração não existir.
    """
    def __init__(self):
        self.states = defaultdict(complex)

    def set(self, config, amplitude):
        """
        Define manualmente a amplitude para uma configuração específica.

        Args:
            config (tuple): A configuração da MTQ (estado, cabeça, fita) para a qual
                            a amplitude será definida.
            amplitude (complex): O valor complexo da amplitude.
        """
        self.states[config] = amplitude

    def normalize(self):
        """
        Normaliza as amplitudes para que a soma dos quadrados de seus módulos seja 1 (||ψ|| = 1),
        condição necessária em mecânica quântica.

        Se a norma for zero (e.g., todos os estados têm amplitude zero), a normalização é ignorada
        para evitar divisão por zero.
        """

        # Calcula a norma euclidiana do vetor de estado (raiz quadrada da soma dos quadrados dos módulos)
        norm = np.sqrt(sum(abs(amp)**2 for amp in self.states.values()))
        if norm == 0:
            return
        # Divide cada amplitude pela norma para normalizar o vetor de estado
        for key in self.states:
            self.states[key] /= norm

    def measure(self):
        """
        Mede o sistema: colapsa o vetor de estado para uma única configuração
        com probabilidade proporcional ao quadrado da amplitude (regra de Born).

        A regra de Born estabelece que a probabilidade de observar um estado é
        o quadrado do módulo de sua amplitude complexa.

        Returns:
            tuple: A configuração (estado, cabeça, fita) que foi medida.

        Raises:
            RuntimeError: Se não houver configurações no registrador para medir,
                          indicando um possível erro na execução da máquina.
        """
        # Calcula a soma total dos quadrados das amplitudes (deve ser 1 se normalizado corretamente)
        total = sum(abs(amp)**2 for amp in self.states.values())
        r = random.random() * total
        acc = 0
        for config, amp in sorted(self.states.items(), key=lambda x: -abs(x[1])**2):
            acc += abs(amp)**2
            if acc >= r:
                return config
        if not self.states:
            # Caso em que o registrador está vazio ou todas as amplitudes são zero
            raise RuntimeError(
            "[Erro] Nenhuma configuração final foi alcançada.\n"
            "→ Possíveis causas:\n"
            "- Entrada inválida ou inconsistente com as transições.\n"
            "- Número de passos insuficiente (aumente 'max_steps').\n"
            "- Transições não levam ao(s) estado(s) final(is).\n\n"
            "Sugestão: execute visualize_amplitudes() para inspecionar os caminhos gerados."
        )

        return random.choice(list(self.states.keys()))

    def apply_unitary(self, U):
        """
        Aplica uma "matriz unitária" esparsa (representada como dicionário) ao vetor de estado.
        O resultado é um novo vetor de estado (novo |ψ⟩ = U |ψ⟩).

        Em computação quântica, a evolução do estado é sempre por operadores unitários.
        Aqui, 'U' é um dicionário que mapeia (estado_origem, estado_destino) para o valor complexo
        do elemento da matriz unitária correspondente.

        Args:
            U (dict): Um dicionário na forma { (from_state, to_state): valor_complexo }
                      representando o operador unitário esparso.
        """
        new_states = defaultdict(complex)

        # Itera sobre os elementos do operador unitário
        for (from_state, to_state), matrix_value in U.items():
            if from_state in self.states:
                new_states[to_state] += self.states[from_state] * matrix_value
        self.states = new_states
        self.normalize()

# -----------------------------------------------------------------------------
# 2. QuantumTape: representa uma fita em superposição de conteúdos
# -----------------------------------------------------------------------------
class QuantumTape:
    """
    A fita quântica mantém múltiplos conteúdos simultaneamente em superposição.

    Cada possível conteúdo da fita (representado como uma tupla de símbolos)
    possui uma amplitude complexa associada, permitindo a exploração paralela
    de diferentes estados da fita.

    { conteúdo_da_fita (tuple): amplitude_complexa }

    Exemplo:
        {'10101': 0.7, '11101': 0.3} → sobreposição de fitas

    Atributos:
        superposed (defaultdict): Um dicionário onde as chaves são as tuplas dos
                                   conteúdos da fita e os valores são suas
                                   amplitudes complexas.
    """
    def __init__(self, possible_contents):
        self.superposed = defaultdict(complex)
        for content in possible_contents:
            self.superposed[tuple(content)] = 1.0 + 0j  # Inicializa com amplitude 1
        self.normalize()

    def normalize(self):
        """
        Normaliza as amplitudes para que a soma dos quadrados das amplitudes
        seja igual a 1 (condição de normalização para a fita).
        """
        norm = np.sqrt(sum(abs(a)**2 for a in self.superposed.values()))
        if norm == 0:
            return
        for k in self.superposed:
            self.superposed[k] /= norm

    def measure(self):
        """
        Mede a fita, retornando um dos conteúdos de fita possíveis,
        selecionado probabilisticamente de acordo com sua probabilidade
        (|amplitude|^2).

        Returns:
            tuple: O conteúdo da fita (tupla de símbolos) que foi medido.
        """
        items = list(self.superposed.items())
        probs = [abs(a)**2 for (_, a) in items]
        choice = random.choices(items, weights=probs, k=1)[0]
        return choice[0]  # Retorna apenas o conteúdo da fita

# -----------------------------------------------------------------------------
# 3. Operadores Quânticos (Oracle, Difusão, Decoerência)
# -----------------------------------------------------------------------------
def oracle_operator(marked_state):
    """
    Gera um operador de oráculo que aplica uma inversão de fase (-1)
    no estado marcado (target state).

    Este é um componente chave em algoritmos como o de Grover, onde o oráculo
    identifica a solução e a "marca" com uma fase negativa para que o
    operador de difusão possa amplificar sua amplitude.

    Args:
        marked_state (tuple): A configuração da MTQ que deve ser "marcada" pelo oráculo.

    Returns:
        callable: Uma função `op(qreg)` que, quando chamada, aplica a inversão de fase
                  à amplitude da `marked_state` dentro do `QuantumRegister`.
    """
    def op(qreg):
        for state in qreg.states:
            if state == marked_state:
                qreg.states[state] *= -1  # Inversão de fase
    return op

def diffusion_operator(qreg):
    """
    Implementa o operador de difusão de Grover: reflete todas as amplitudes
    dos estados no `QuantumRegister` em torno da amplitude média.

    Este operador, em conjunto com o oráculo, é o cerne da amplificação de Grover.
    Ele amplifica as amplitudes dos estados marcados pelo oráculo (que agora têm
    fase invertida) e diminui as amplitudes dos demais estados.

    Args:
        qreg (QuantumRegister): O registrador quântico ao qual o operador de difusão
                                será aplicado.
    """
    n = len(qreg.states)
    mean_amp = sum(qreg.states.values()) / n
    for k in qreg.states:
        qreg.states[k] = 2 * mean_amp - qreg.states[k]

def apply_decoherence(qreg, prob_error=0.05):
    """
    Simula a decoerência (ruído quântico) em um `QuantumRegister`.

    Com uma certa probabilidade (`prob_error`), a fase complexa de uma amplitude
    é removida, restando apenas seu módulo (amplitude real positiva). Isso simula
    a perda de coerência quântica devido à interação com o ambiente.

    Args:
        qreg (QuantumRegister): O registrador quântico ao qual a decoerência será aplicada.
        prob_error (float): A probabilidade (entre 0 e 1) de ocorrer um erro de decoerência
                            para cada estado no registrador.
    """
    for state in qreg.states:
        if random.random() < prob_error:
            qreg.states[state] = abs(qreg.states[state]) + 0j  # Remove fase
    qreg.normalize()
