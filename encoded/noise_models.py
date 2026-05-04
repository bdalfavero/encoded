from typing import Dict
import cirq

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


if __name__ == "__main__":
    qs = cirq.LineQubit.range(3)
    noise_map = {q: cirq.DepolarizingChannel(p=1. / (q.x + 1)).on(q) for q in qs}
    noise_gate = QubitDependentNoiseGate(noise_map)
    print(noise_gate.num_qubits())
    circuit = cirq.Circuit()
    for q in qs:
        circuit.append(noise_gate.on(q))
    print(circuit)