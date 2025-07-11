"""
main.py

Simulador de Máquina de Turing Quântica (MTQ), com suporte à evolução adaptativa
e visualização de amplitudes. Permite medir estados finais e registrar log de
probabilidades ao longo dos passos.

Autor: Emanuel Lopes Silva
Data: Julho de 2025
"""
import cmath
from quantum_turing_machine import QuantumTuringMachine
from copy import deepcopy
def transitions_with_phase(raw_transitions):
    """Adiciona fase complexa (1.0+0j) a cada transição."""
    with_phase = {}
    for key, trans_list in raw_transitions.items():
        with_phase[key] = [(ns, sym, dir, 1.0 + 0j) for (ns, sym, dir) in trans_list]
    return with_phase

# Transições base
transitions = {
    ('q0', '0'): [('q1', '0', 'D')],
    ('q1', 'a'): [('q2', 'X', 'D')],
    ('q1', 'b'): [('q3', 'X', 'D')],
    ('q1', 't'): [('qf', 't', 'E')],
    ('q2', 'a'): [('q2', 'a', 'D'), ('q4', 'Y', 'E')],
    ('q2', 'b'): [('q2', 'b', 'D')],
    ('q3', 'a'): [('q3', 'a', 'D')],
    ('q3', 'b'): [('q3', 'b', 'D'), ('q4', 'Y', 'E')],
    ('q4', 'a'): [('q4', 'a', 'E')],
    ('q4', 'b'): [('q4', 'b', 'E')],
    ('q4', 'X'): [('q5', 'X', 'D')],
    ('q5', 'a'): [('q6', 'X', 'D')],
    ('q5', 'b'): [('q8', 'X', 'D')],
    ('q5', 'Y'): [('q11', 'Y', 'D')],
    ('q6', 'a'): [('q6', 'a', 'D')],
    ('q6', 'b'): [('q6', 'b', 'D')],
    ('q6', 'Y'): [('q7', 'Y', 'D')],
    ('q7', 'Y'): [('q7', 'Y', 'D')],
    ('q7', 'a'): [('q10', 'Y', 'E')],
    ('q8', 'a'): [('q8', 'a', 'D')],
    ('q8', 'b'): [('q8', 'b', 'D')],
    ('q8', 'Y'): [('q9', 'Y', 'D')],
    ('q9', 'Y'): [('q9', 'Y', 'D')],
    ('q9', 'b'): [('q10', 'Y', 'E')],
    ('q10', 'a'): [('q10', 'a', 'E')],
    ('q10', 'b'): [('q10', 'b', 'E')],
    ('q10', 'Y'): [('q10', 'Y', 'E')],
    ('q10', 'X'): [('q5', 'X', 'D')],
    ('q11', 'Y'): [('q11', 'Y', 'D')],
    ('q11', 't'): [('qf', 't', 'E')],
}

import json
from pathlib import Path

log_amplitudes = []  # Lista global para armazenar o log por passo

def run_until_final_state(qtm, max_steps_limit=100):
    """
    Executa a MTQ até atingir um estado final ou o limite máximo de passos.

    Args:
        qtm (QuantumTuringMachine): Instância da máquina quântica.
        max_steps_limit (int): Máximo de passos permitidos.
    """
    print(f"\n\033[95m Iniciando busca adaptativa por estado final...\033[0m")
    steps = 2
    found = False
    historico_passos = []

    while steps <= max_steps_limit:
        print(f"\n\033[97m Tentando com {steps} passos...\033[0m")
        try:
            qtm.reset()
            qtm.run(max_steps=steps)

            snapshot = deepcopy(qtm.register.states)
            historico_passos.append((steps, snapshot))

            # ===  Registro estruturado em log_amplitudes ===
            step_log = []
            for (tape, head, state), amp in snapshot.items():
                step_log.append({
                    "passo": steps,
                    "estado": state,
                    "cabeca": head,
                    "fita": ''.join(tape),
                    "amplitude_real": amp.real,
                    "amplitude_imag": amp.imag,
                    "probabilidade": abs(amp)**2
                })
            log_amplitudes.append(step_log)

            qtm.visualize_amplitudes()

            estados_finais = [state for (_, _, state) in qtm.register.states.keys()]
            if any(s in qtm.final_states for s in estados_finais):
                print(f"\n\033[92m Estado final alcançado com {steps} passos.\033[0m")
                found = True
                break

        except RuntimeError as e:
            print(f"\033[91m XX Erro durante execução no passo {steps}: {e}\033[0m")
            historico_passos.append((steps, {}))
            log_amplitudes.append([])  # Mantém alinhamento com os passos
            steps += 1
            continue

        steps += 1

    if not found:
        print(f"\n\033[91m ! Nenhum estado final encontrado até {max_steps_limit} passos.\033[0m")

    print(f"\n\033[96m--- Evolução Quântica ---\033[0m")
    for step, estados in historico_passos:
        print(f"\n\033[93m Passo {step}:\033[0m")
        for (tape, head, state), amp in sorted(estados.items(), key=lambda x: -abs(x[1])**2):
            prob = abs(amp)**2
            if prob > 0.001:
                print(f"  Estado: {state:>3} | Cabeça: {head:>2} | Fita: {''.join(tape)} | Prob: {prob:.4f}")

    try:
        resultado = qtm.measure()
        tape, head, state = resultado
    except RuntimeError as e:
        print(f"\033[91m Medida falhou: {e}\033[0m")

    # ===  Salva o log no final ===
    output_path = Path("log_amplitudes.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(log_amplitudes, f, indent=2, ensure_ascii=False)
    print(f"\n\033[92m✔ Log de amplitudes salvo em: {output_path.resolve()}\033[0m")

#O código abaixo foi comentado porque a execução adaptativa por estado final geraria muitos logs ao mesmo tempo, o que pode tornar difícil o entenddimento e a visualização do resultado final.
# Se necessário, pode ser descomentado para testes específicos. Ele funciona legal, recomendo ver depois.


#def run_until_final_state(qtm, max_steps_limit=100):
   # print(f"\n \033[95m Iniciando busca adaptativa por estado final...\033[0m")
  #  steps = 2
  #  found = False

   # while steps <= max_steps_limit:
   #     print(f"\n \033[97m Tentando com {steps} passos...\033[0m")
    #    try:
     #       qtm.reset()  # reinicia o registrador para simulação limpa
    #        qtm.run(max_steps=steps)

            # Visualiza os caminhos com suas amplitudes a cada passo
      #      qtm.visualize_amplitudes()

            # Verifica se algum estado final foi atingido
       #     estados = [state for (_, _, state) in qtm.register.states.keys()]
       #     if any(s in qtm.final_states for s in estados):
       #         print(f"\n \033[92m Estado final alcançado com {steps} passos.\033[0m")
       #         qtm.measure()  # Apenas mede se final foi encontrado
       #         found = True
       #         break

     #   except RuntimeError as e:
       #     print(f"\033[91m XX Erro durante execução: {e}\033[0m")
 #           break
#
 #       steps += 1
#
 #   if not found:
#        print(f"\n ! Nenhum estado final encontrado até {max_steps_limit} passos.")


# Só sai o resultado final
#try:
   # qtm.run(max_steps=int(numero_passos))
#except RuntimeError as e:
  #  print(e)
  #  exit()
#if not any(state in final_states for (state, _, _) in qtm.register.states):
   # print(" Nenhum estado final foi alcançado. Tente aumentar max_steps ou revisar a entrada.")
# Visualiza todos os caminhos e amplitudes
#qtm.visualize_amplitudes()


# ⬇ Código principal aqui dentro
if __name__ == "__main__":
    print(f"\n\033[1;32m Iniciando execução quântica...\033[0m")
    tape_input = input("Digite uma cadeia para testar na MTQ (ex: 0ababt): ").strip()
    transitions_phased = transitions_with_phase(transitions)
    final_states = ['qf']
    qtm = QuantumTuringMachine(
        transitions=transitions_phased,
        initial_state='q0',
        input_tape=tape_input,
        final_states=final_states
    )
    auto = input("Deseja calcular automaticamente o número de passos? (s/n): ").strip().lower()
    if auto == "s":
        k = 4  # fator de multiplicação, ajustável conforme a complexidade da máquina
        b = 10 # offset de segurança
        numero_passos = k * len(tape_input) + b
        print(f"max_steps estimado automaticamente: {numero_passos}")
    else:
        try:
            numero_passos = int(input("Digite o número de passos (max_steps) para a execução: ").strip())
        except ValueError:
            print("Valor inválido para número de passos.")
            exit()

    run_until_final_state(qtm, max_steps_limit=numero_passos)