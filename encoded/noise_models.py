from typing import Dict
import stim
import cirq
import stimcirq

class QubitDependentNoiseGate(cirq.Gate):

    def __init__(self, qubit_map: Dict[cirq.Qid, cirq.Operation]):
        super(QubitDependentNoiseGate, self)
        self._qubit_map = qubit_map
    
    def _num_qubits_(self) -> int:
        return 1
    
    def _circuit_diagram_info_(self, args):
        return [f"Noise"] * self._num_qubits_()
    
    def _decompose_(self, qubits):
        for qubit in qubits:
            yield self._qubit_map[qubit]


def stim_circuit_with_qubit_dependent_noise_rate(stim_ckt: stim.Circuit, noise_map: Dict[int, float]) -> stim.Circuit:
    """Add depolarizing noise to a circuit with a different rate for each qubit."""

    qubit_map = {}
    for idx, noise_rate in noise_map.items():
        q = cirq.LineQubit(idx)
        qubit_map[q] = cirq.DepolarizingChannel(p=noise_rate).on(q)
    noise_gate = QubitDependentNoiseGate(qubit_map)
    cirq_circuit = stimcirq.stim_circuit_to_cirq_circuit(stim_ckt)
    noisy_cirq_circuit = cirq_circuit.with_noise(noise_gate)
    noisy_stim_circuit = stimcirq.cirq_circuit_to_stim_circuit(noisy_cirq_circuit)
    return noisy_stim_circuit


if __name__ == "__main__":
    stim_ckt = stim.Circuit()
    stim_ckt.append("H", 0)
    stim_ckt.append("X", 1)
    noise_map = {0: 0.1, 1: 0.2}
    noisy_ckt = stim_circuit_with_qubit_dependent_noise_rate(stim_ckt, noise_map)
    print(noisy_ckt)