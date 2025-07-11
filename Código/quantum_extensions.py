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

    O estado da máquina é representado por:
    { (estado, cabeça, fita): amplitude complexa }

    Métodos:
        - set: define amplitude manualmente.
        - normalize: ajusta para ||ψ|| = 1.
        - measure: colapsa conforme probabilidade.
        - apply_unitary: aplica operador unitário (Grover, etc).

    """
    def __init__(self):
        self.states = defaultdict(complex)

    def set(self, config, amplitude):
        """
        Define manualmente a amplitude para uma configuração.
        """
        self.states[config] = amplitude

    def normalize(self):
        """
        Normaliza as amplitudes para que a soma dos quadrados seja 1 (||ψ|| = 1),
        condição necessária em mecânica quântica.
        """
        norm = np.sqrt(sum(abs(amp)**2 for amp in self.states.values()))
        if norm == 0:
            return
        for key in self.states:
            self.states[key] /= norm

    def measure(self):
        """
        Mede o sistema: retorna uma configuração com probabilidade proporcional
        ao quadrado da amplitude (regra de Born).
        """
        total = sum(abs(amp)**2 for amp in self.states.values())
        r = random.random() * total
        acc = 0
        for config, amp in sorted(self.states.items(), key=lambda x: -abs(x[1])**2):
            acc += abs(amp)**2
            if acc >= r:
                return config
        if not self.states:
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
        Aplica uma "matriz unitária" esparsa representada como dicionário:
        U: { (from_state, to_state): valor_complexo }
        Resultado: novo vetor de estado = U |ψ⟩
        """
        new_states = defaultdict(complex)
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
    A fita quântica mantém múltiplos conteúdos simultaneamente:
    { conteúdo_da_fita (tuple): amplitude_complexa }

    Exemplo:
        {'10101': 0.7, '11101': 0.3} → sobreposição de fitas
    """
    def __init__(self, possible_contents):
        self.superposed = defaultdict(complex)
        for content in possible_contents:
            self.superposed[tuple(content)] = 1.0 + 0j  # Inicializa com amplitude 1
        self.normalize()

    def normalize(self):
        """
        Normaliza as amplitudes para que a soma dos quadrados das amplitudes
        seja igual a 1.
        """
        norm = np.sqrt(sum(abs(a)**2 for a in self.superposed.values()))
        if norm == 0:
            return
        for k in self.superposed:
            self.superposed[k] /= norm

    def measure(self):
        """
        Mede a fita, retornando uma das fitas possíveis proporcionalmente
        à sua probabilidade.
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
    Gera um operador que aplica uma inversão de fase no estado marcado.
    Isso é usado em Grover para reforçar a probabilidade desse estado.

    Exemplo: se o estado |x⟩ é o desejado, então aplica:
        |x⟩ → -|x⟩
    """
    def op(qreg):
        for state in qreg.states:
            if state == marked_state:
                qreg.states[state] *= -1  # Inversão de fase
    return op

def diffusion_operator(qreg):
    """
    Implementa o operador de difusão de Grover:
    reflete todos os estados em torno da média.
    Amplifica amplitudes dos estados marcados pelo oráculo.
    """
    n = len(qreg.states)
    mean_amp = sum(qreg.states.values()) / n
    for k in qreg.states:
        qreg.states[k] = 2 * mean_amp - qreg.states[k]

def apply_decoherence(qreg, prob_error=0.05):
    """
    Simula decoerência (ruído quântico): com certa probabilidade,
    remove a fase complexa da amplitude (como uma perda de coerência).
    Isso aproxima a simulação do que ocorre em sistemas reais ruidosos.
    """
    for state in qreg.states:
        if random.random() < prob_error:
            qreg.states[state] = abs(qreg.states[state]) + 0j  # Remove fase
    qreg.normalize()
