import cmath
import math
from collections import defaultdict
from quantum_extensions import QuantumRegister, QuantumTape, diffusion_operator, oracle_operator, apply_decoherence

def phase(): return cmath.exp(1j * math.pi)  # Fase -1

class QuantumTuringMachine:
    def __init__(self, transitions, initial_state, input_tape, final_states):
        self.transitions = transitions  # dicionário: (estado, simbolo) -> [(novo_estado, novo_simbolo, direcao, fase)]
        self.initial_state = initial_state
        self.initial_tape = QuantumTape([list(input_tape)])  # Fita quântica real
        self.register = QuantumRegister()
        for tape in self.initial_tape.superposed:
            self.register.set((tuple(tape), 0, initial_state), 1.0 + 0j)
        self.register.normalize()
        self.final_states = set(final_states)
        self.step_count = 0
    def phase(): return cmath.exp(1j * math.pi)  # Fase -1
    def reset(self):
        self.register = QuantumRegister()
        for tape in self.initial_tape.superposed:
            self.register.set((tuple(tape), 0, self.initial_state), 1.0 + 0j)
        self.register.normalize()
        self.step_count = 0
    def step(self, decohere=False):
        next_register = QuantumRegister()

        for (tape, head, state), amplitude in self.register.states.items():
            if head < 0 or head >= len(tape):
                continue

            symbol = tape[head]
            actions = self.transitions.get((state, symbol), [])

            if not actions:
                continue

            num_actions = len(actions)
            for (new_state, new_symbol, direction, phase) in actions:
                new_tape = list(tape)
                new_tape[head] = new_symbol
                new_head = head + 1 if direction == "D" else head - 1

                if new_head < 0 or new_head >= len(new_tape):
                    continue

                new_config = (tuple(new_tape), new_head, new_state)

                # ⚠️ Corrigido: divide amplitude entre transições e aplica fase
                split_amplitude = amplitude / (num_actions**0.5)
                next_register.states[new_config] += split_amplitude * phase

        if decohere:
            apply_decoherence(next_register, prob_error=0.1)

        next_register.normalize()
        self.register = next_register
        self.step_count += 1

    def run(self, max_steps=10, oracle=None, apply_diffusion=False):
        for _ in range(max_steps):
            if oracle:
                oracle(self.register)
            if apply_diffusion:
                diffusion_operator(self.register)
            self.step()

        return self.measure()

    def measure(self):
        config = self.register.measure()
        tape, head, state = config
        print("\n\033[94mResultado da Medida:\033[0m")
        print("Estado final:", state)
        print("Fita final:", ''.join(tape))
        print("Posição da cabeça:", head)
        return config

    def visualize_amplitudes(self):
        print("\n--- Amplitudes Finais ---")
        sorted_states = sorted(
            self.register.states.items(),
            key=lambda item: abs(item[1])**2,
            reverse=True
        )
        for (tape, head, state), amp in sorted_states:
            prob = round(abs(amp)**2, 4)
            if prob > 0.001:
                print(f"Estado: {state:>3} | Cabeça: {head:>2} | Fita: {''.join(tape)} | Amplitude: {amp:.4f} | Prob: {prob:.4f}")