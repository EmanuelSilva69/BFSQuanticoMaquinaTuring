"""
quantum_turing_machine.py

Implementação de uma Máquina de Turing Quântica (MTQ), com suporte a:
- Superposição de estados e fitas
- Transições com fase complexa
- Medição conforme a regra de Born
- Oráculo (Grover), difusão e decoerência
- Visualização das amplitudes finais

Autor: Emanuel Lopes Silva
Data: Julho de 2025
"""
import cmath
import math
from collections import defaultdict
from quantum_extensions import QuantumRegister, QuantumTape, diffusion_operator, oracle_operator, apply_decoherence
"""
    Retorna a fase quântica -1 (e^{iπ}).

    Esta função é um helper para definir uma fase comum em operações quânticas,
    equivalente à multiplicação por -1 na amplitude. A fase de pi (π)
    corresponde a e^(iπ) = cos(π) + i*sin(π) = -1 + 0i = -1.

    Returns:
        complex: valor complexo correspondente a -1.
"""
def phase(): return cmath.exp(1j * math.pi)  # Fase -1

class QuantumTuringMachine:
    """
    Classe que representa uma Máquina de Turing Quântica (MTQ),
    com suporte a transições não determinísticas, amplitudes complexas
    e visualização da evolução dos estados.

    A máquina evolui seu registrador de estados em paralelo,
    aplicando fases e normalizando a cada passo.

    Atributos:
        transitions (dict): dicionário de transições no formato
                            (estado_atual, símbolo_lido) → [(novo_estado, novo_símbolo, direção, fase)]
                            Cada tupla representa uma possível transição.
        initial_state (str): estado inicial da máquina.
        initial_tape (QuantumTape): fita inicial representada em superposição,
                                    permitindo que múltiplos conteúdos sejam considerados.
        register (QuantumRegister): vetor de estado da máquina, que armazena as
                                    amplitudes complexas de todas as configurações em superposição.
        final_states (set): conjunto de estados de aceitação da máquina.
        step_count (int): contador de passos executados pela simulação.
    """
    def __init__(self, transitions, initial_state, input_tape, final_states):
        self.transitions = transitions  # dicionário: (estado, simbolo) -> [(novo_estado, novo_simbolo, direcao, fase)]
        self.initial_state = initial_state

        # Inicializa a fita quântica com o conteúdo da fita de entrada.
        # A fita é envolvida em uma lista para que QuantumTape possa lidar com ela
        # como um dos "conteúdos possíveis".
        self.initial_tape = QuantumTape([list(input_tape)])  # Fita quântica real
        self.register = QuantumRegister()

        # Inicializa o registrador de estados com a configuração inicial.
        # Cada possível "fita" inicial (no caso simples, apenas uma) é usada
        # para criar uma configuração inicial na posição 0 com o estado inicial
        # e amplitude 1.0 (probabilidade 1).
        for tape in self.initial_tape.superposed:
            self.register.set((tuple(tape), 0, initial_state), 1.0 + 0j)
        # Normaliza o registrador para garantir que as amplitudes somem 1.
        self.register.normalize()
        self.final_states = set(final_states)
        self.step_count = 0

    # Redefinição da função phase(), que não seria necessária aqui, pois já foi definida globalmente.
    # Pode ser um resquício de desenvolvimento.
    def phase(): return cmath.exp(1j * math.pi)  # Fase -1

    def reset(self):
        """
        Reinicializa o registrador quântico e o contador de passos.

        Isso permite reexecutar a MTQ a partir de seu estado inicial sem
        manter informações de execuções anteriores no registrador.
        """
        self.register = QuantumRegister()
        for tape in self.initial_tape.superposed:
            self.register.set((tuple(tape), 0, self.initial_state), 1.0 + 0j)
        self.register.normalize()
        self.step_count = 0
    
    def step(self, decohere=False):
        """
        Executa um único passo de evolução da MTQ.

        Em um passo, a máquina processa todas as configurações atualmente em superposição,
        aplica as transições e atualiza as amplitudes. Se houver múltiplas ações para
        uma mesma (estado, símbolo), a amplitude é dividida entre elas e a fase é aplicada.

        Args:
            decohere (bool): Se True, aplica um operador de decoerência (ruído quântico)
                             ao registrador após a aplicação das transições.
                             Isso simula a perda de coerência do sistema.
        """
        next_register = QuantumRegister()# Cria um novo registrador para o próximo passo

        # Itera sobre cada configuração (tape, head, state) e sua amplitude atual
        for (tape, head, state), amplitude in self.register.states.items():
            # Se a amplitude for zero, ignora essa configuração
            if head < 0 or head >= len(tape):
                continue

            symbol = tape[head]# Lê o símbolo atual na posição da cabeça

            # Busca as ações possíveis para (estado, símbolo) na tabela de transições
            actions = self.transitions.get((state, symbol), [])

            # Se não houver ações, esta configuração não evolui e é ignorada
            if not actions:
                continue

            num_actions = len(actions)# Número de transições possíveis para esta configuração

            # Para cada ação possível, cria uma nova configuração
            for (new_state, new_symbol, direction, phase) in actions:
                new_tape = list(tape)
                new_tape[head] = new_symbol
                new_head = head + 1 if direction == "D" else head - 1

                if new_head < 0 or new_head >= len(new_tape):
                    continue

                new_config = (tuple(new_tape), new_head, new_state)

                # Corrigido: divide amplitude entre transições e aplica fase
                split_amplitude = amplitude / (num_actions**0.5)
                next_register.states[new_config] += split_amplitude * phase

        # Se a decoerência estiver ativada, aplica ruído quântico
        if decohere:
            apply_decoherence(next_register, prob_error=0.1)

        next_register.normalize() # Normaliza o novo registrador após todas as transições
        self.register = next_register # Atualiza o registrador da máquina para o próximo estado
        self.step_count += 1 # Incrementa o contador de passos

    def run(self, max_steps=10, oracle=None, apply_diffusion=False):
        """
        Executa a MTQ por um número fixo de passos.

        Durante a execução, pode aplicar operadores de oráculo e difusão
        (inspirados no algoritmo de Grover) a cada passo, se especificado.

        Args:
            max_steps (int): número máximo de passos a serem executados.
            oracle (callable, opcional): função de oráculo a ser aplicada a cada passo.
                                         Se fornecida, modifica o registrador diretamente.
            apply_diffusion (bool): Se True, aplica o operador de difusão de Grover
                                    após o oráculo em cada passo.

        Returns:
            tuple: configuração final (tape, head, state) após a medição no final da execução.
        """
        for _ in range(max_steps):
            # Aplica o operador de oráculo se definido
            if oracle:
                oracle(self.register)
            # Aplica o operador de difusão se ativado
            if apply_diffusion:
                diffusion_operator(self.register)
            self.step() # Executa um passo da MTQ

        return self.measure()

    def measure(self):
        """
        Realiza uma medição sobre o registrador, colapsando a superposição
        para uma única configuração observável.

        A seleção da configuração colapsada é probabilística, baseada na
        regra de Born (probabilidade = |amplitude|^2).

        Returns:
            tuple: A configuração (tape, head, state) para a qual o sistema colapsou.
        Raises:
            RuntimeError: Se nenhuma configuração final foi alcançada ou o registrador está vazio.
        """
        config = self.register.measure()
        tape, head, state = config
        print("\n\033[94mResultado da Medida:\033[0m")
        print("Estado final:", state)
        print("Fita final:", ''.join(tape))
        print("Posição da cabeça:", head)
        return config

    def visualize_amplitudes(self):
        """
        Exibe no terminal os estados mais prováveis do registrador atual.

        Lista as configurações (fita, cabeça, estado) com suas amplitudes e
        probabilidades (|amplitude|^2), ordenadas da mais provável para a menos.
        Filtra configurações com probabilidade muito baixa para uma saída mais limpa.
        """
        # Ordena os estados do registrador pela probabilidade (módulo da amplitude ao quadrado)
        print("\n--- Amplitudes Finais ---")
        sorted_states = sorted(
            self.register.states.items(),
            key=lambda item: abs(item[1])**2,
            reverse=True
        )
        # Imprime apenas os estados com probabilidade acima de um certo limiar
        for (tape, head, state), amp in sorted_states:
            prob = round(abs(amp)**2, 4)
            if prob > 0.001:
                print(f"Estado: {state:>3} | Cabeça: {head:>2} | Fita: {''.join(tape)} | Amplitude: {amp:.4f} | Prob: {prob:.4f}")