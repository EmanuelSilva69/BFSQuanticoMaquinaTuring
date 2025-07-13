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
    Simulação de uma BFS (Busca em Largura) com reforço quântico inspirado
    no algoritmo de Grover.

    Esta classe demonstra como os princípios de oráculo e amplificação de
    amplitude de Grover podem ser aplicados em uma busca clássica para
    priorizar caminhos que levam a um estado desejado. Não é uma simulação
    de uma Máquina de Turing Quântica em si, mas uma inspiração híbrida.

    Atributos:
        tm (Any): Objeto que representa a máquina de Turing (ou MTQ) usada
                  para simulação das transições. Espera-se que possua um
                  atributo `states` que armazene as regras de transição.
        max_iterations (int): Número máximo de iterações de reforço (amplificação).
                              Quanto mais iterações, maior o reforço dos caminhos.
        final_state (str): O estado da máquina de Turing considerado como o
                           estado de aceitação ou objetivo da busca.
    """

    def __init__(self, tm: Any):
        """
        Inicializa a simulação com a máquina de Turing fornecida.

        Args:
            tm (Any): Objeto que representa a máquina de Turing com atributo 'states'.
                      Deve ter um dicionário de transições para simular os passos.
        """
        self.tm = tm
        self.max_iterations = 5  # Número de iterações de Grover inspiradas
        self.final_state = 'qf' # Estado objetivo que o oráculo irá "marcar"

    def oracle(self, config: Tuple[tuple, int, str]) -> int:
        """
        Oráculo quântico simulado: retorna +1 para configurações que incluem
        o estado final ('qf'), e -1 para as demais.

        No algoritmo de Grover, o oráculo "marca" os estados solução invertendo
        sua fase. Aqui, simulamos isso retornando -1 para a amplitude.

        Args:
            config (tuple): Configuração atual da máquina (fita, posição da cabeça, estado interno).

        Returns:
            int: -1 se o estado da configuração for igual a `self.final_state`,
                 +1 caso contrário.
        """
        _, _, state = config # Desempacota a tupla de configuração para obter o estado
        return 1 if state == self.final_state else -1 # Retorna -1 se for o estado final

    def diffusion(self, amplitudes: Dict[Tuple, float]) -> Dict[Tuple, float]:
        """
        Aplica o operador de difusão de Grover, simulando o espelhamento das
        amplitudes em torno da média.

        Este operador amplifica as amplitudes dos caminhos "marcados" pelo oráculo
        e rebaixa as amplitudes dos caminhos não marcados, aumentando a probabilidade
        de encontrar a solução. É uma operação fundamental de "amplificação de amplitude".

        Args:
            amplitudes (dict): Dicionário de amplitudes (valores reais aqui,
                                para simplificação) associadas a cada configuração.
                                As chaves são as configurações da máquina e os valores
                                são suas amplitudes.

        Returns:
            dict: Novo dicionário de amplitudes após a aplicação do operador de difusão.
        """
        total = sum(amplitudes.values())

        # Calcula a amplitude média
        mean = total / len(amplitudes)
        for k in amplitudes:
            # A nova amplitude é calculada como 2 * (média) - (amplitude antiga)
            amplitudes[k] = 2 * mean - amplitudes[k]
        return amplitudes

    def run(self, input_string: str) -> None:
        """
        Executa a simulação BFS com reforço quântico inspirado por Grover.

        A busca em largura explora as configurações da máquina camada por camada.
        Em cada iteração, as amplitudes são ajustadas pelo oráculo e difusão
        para concentrar a "probabilidade" nos caminhos que levam ao estado final.

        Args:
            input_string (str): Cadeia de entrada (sequência de símbolos) a ser
                                 processada pela máquina simulada.
        """
        # Configuração inicial
        initial_config = (tuple(input_string), 0, 'q0')
        queue = deque([(initial_config, [initial_config])])
        amplitudes = defaultdict(lambda: 1.0)


        # Loop principal de iterações de amplificação de Grover
        for iteration in range(self.max_iterations):
            next_queue = deque()
            new_amplitudes = defaultdict(float)

            print(f"\n🔁 Iteração {iteration + 1}")

            # Processa todas as configurações da fila atual (BFS)
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