from collections import deque, defaultdict
import copy

class QuantumInspiredBFS:
    def __init__(self, tm):
        self.tm = tm
        self.max_iterations = 5
        self.final_state = 'qf'

    def oracle(self, config):
        """
        Oráculo: retorna +1 para caminhos válidos (estado final), -1 para os demais.
        Em Grover real, é uma inversão de fase. Aqui é ponderação.
        """
        _, _, state = config
        return 1 if state == self.final_state else -1

    def diffusion(self, amplitudes):
        """
        Simulação da difusão (espelhamento em torno da média).
        Amplifica caminhos válidos, reduz inválidos.
        """
        total = sum(amplitudes.values())
        mean = total / len(amplitudes)
        for k in amplitudes:
            # Espelhamento em torno da média (Grover Diffusion Operator)
            amplitudes[k] = 2 * mean - amplitudes[k]
        return amplitudes

    def run(self, input_string):
        # Inicializa configuração
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

                # Aplica oráculo
                amp *= self.oracle(config)

                # Transições possíveis
                actions = self.tm.states.get((state, tape[head] if 0 <= head < len(tape) else ' '), set())

                for new_state, write_sym, move in actions:
                    new_tape = list(tape)
                    new_tape[head] = write_sym
                    new_head = head + (1 if move == 'D' else -1)

                    if not (0 <= new_head < len(new_tape)):
                        continue

                    new_config = (tuple(new_tape), new_head, new_state)
                    new_path = path + [new_config]

                    # Soma da amplitude para esse novo caminho
                    new_amplitudes[new_config] += amp
                    next_queue.append((new_config, new_path))

            # Aplica difusão (Grover amplification)
            amplitudes = self.diffusion(new_amplitudes)
            queue = next_queue

        # Após iterações, selecionar o melhor caminho
        best_config = max(amplitudes, key=amplitudes.get)
        best_amp = amplitudes[best_config]

        print("\n✅ Caminho mais provável após simulação:")
        print("Estado final:", best_config[2])
        print("Fita final:", ''.join(best_config[0]))
        print("Posição da cabeça:", best_config[1])
        print("Amplitude:", best_amp)
