"""A code with stabilizer generators {ZZII, IZZI, XXXX}."""

from typing import List, Tuple
import cirq
from cirq import Circuit, Qid
from encoded.repetition_code import encoding_repetition

qs = cirq.LineQubit.range(4)

stabilizer_generators = [
    cirq.Z.on(qs[0]) * cirq.Z.on(qs[1]),
    cirq.Z.on(qs[1]) * cirq.Z.on(qs[2]),
    cirq.X.on(qs[0]) * cirq.X.on(qs[1]) * cirq.X.on(qs[2]) * cirq.X.on(qs[3])
]

# An extra unitary is used for the enocding circuit and the logicals. See Algo II.
U_XXX = cirq.Circuit()
for q in qs[:-1]:
    U_XXX.append(cirq.H(q))
for q in qs[:-1]:
    U_XXX.append(cirq.CNOT(q, qs[-1]))
for q in qs:
    U_XXX.append(cirq.H(q))

z_bar = cirq.Z.on(qs[0]) * cirq.Z.on(qs[3])
x_bar = cirq.X.on(qs[0]) * cirq.X.on(qs[1]) * cirq.X.on(qs[2])

def encoding_expanded_repetition(state: str) -> Tuple[Circuit, List[Qid]]:
    circuit, qubits = encoding_repetition(state, 3)
    qubits.append(cirq.LineQubit(3))
    circuit += U_XXX
    assert circuit.all_qubits() == set(qubits)
    return circuit, qubits