"""
Base.py

Implementa uma simulação inspirada no algoritmo de Grover utilizando busca em largura (BFS).
Aplica um oráculo e um operador de difusão a cada iteração para amplificar caminhos
que levam ao estado de aceitação ('qf').

Autor: Emanuel Lopes Silva
Data: Julho de 2025
"""

from collections import deque, defaultdict
from typing import Tuple, Dict, List, Any
import copy


class QuantumInspiredBFS:
    """
    Simulação de uma BFS com reforço quântico inspirado no algoritmo de Grover.

    Atributos:
        tm: Máquina de Turing usada para simulação (com dicionário de transições).
        max_iterations: Número máximo de iterações de reforço.
        final_state: Estado considerado como aceitação.
    """

    def __init__(self, tm: Any):
        """
        Inicializa a simulação com a máquina de Turing fornecida.

        Args:
            tm (Any): Objeto que representa a máquina de Turing com atributo 'states'.
        """
        self.tm = tm
        self.max_iterations = 5
        self.final_state = 'qf'

    def oracle(self, config: Tuple[tuple, int, str]) -> int:
        """
        Oráculo quântico simulado: retorna +1 para caminhos que levam ao estado final ('qf'),
        e -1 para os demais.

        Args:
            config (tuple): Configuração atual (tape, head, state).

        Returns:
            int: +1 se for estado final, -1 caso contrário.
        """
        _, _, state = config
        return 1 if state == self.final_state else -1

    def diffusion(self, amplitudes: Dict[Tuple, float]) -> Dict[Tuple, float]:
        """
        Aplica o operador de difusão de Grover simulando o espelhamento das amplitudes
        em torno da média.

        Args:
            amplitudes (dict): Dicionário de amplitudes associadas a cada configuração.

        Returns:
            dict: Novo dicionário de amplitudes após a difusão.
        """
        total = sum(amplitudes.values())
        mean = total / len(amplitudes)
        for k in amplitudes:
            amplitudes[k] = 2 * mean - amplitudes[k]
        return amplitudes

    def run(self, input_string: str) -> None:
        """
        Executa a simulação BFS com reforço quântico por Grover.

        Args:
            input_string (str): Cadeia de entrada binária a ser processada.
        """
        # Configuração inicial
        initial_config = (tuple(input_string), 0, 'q0')
        queue = deque([(initial_config, [initial_config])])
        amplitudes = defaultdict(lambda: 1.0)

        for iteration in range(self.max_iterations):
            next_queue = deque()
            new_amplitudes = defaultdict(float)

            print(f"\n🔁 Iteração {iteration + 1}")

            while queue:
                (tape, head, state), path = queue.popleft()
                config = (tape, head, state)
                amp = amplitudes[config]

                # Aplica oráculo (Grover: inversão de fase simulada)
                amp *= self.oracle(config)

                # Obtém transições da MT para esse estado e símbolo
                symbol = tape[head] if 0 <= head < len(tape) else ' '
                actions = self.tm.states.get((state, symbol), set())

                for new_state, write_sym, move in actions:
                    new_tape = list(tape)
                    new_tape[head] = write_sym
                    new_head = head + (1 if move == 'D' else -1)

                    if not (0 <= new_head < len(new_tape)):
                        continue

                    new_config = (tuple(new_tape), new_head, new_state)
                    new_path = path + [new_config]

                    new_amplitudes[new_config] += amp
                    next_queue.append((new_config, new_path))

            # Reforça os caminhos prováveis com difusão (Grover amplification)
            amplitudes = self.diffusion(new_amplitudes)
            queue = next_queue

        # Seleciona o melhor caminho final com maior amplitude
        best_config = max(amplitudes, key=amplitudes.get)
        best_amp = amplitudes[best_config]

        print("\n✅ Caminho mais provável após simulação:")
        print("Estado final:", best_config[2])
        print("Fita final:", ''.join(best_config[0]))
        print("Posição da cabeça:", best_config[1])
        print("Amplitude:", best_amp)