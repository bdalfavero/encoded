import cirq
import stim
from encoded.utils import get_observables, stim_pauli_string_to_cirq, cirq_pauli_string_to_stim

generators = [
    stim.PauliString("ZIZII"),
    stim.PauliString("IZIZI"),
    stim.PauliString("IZZIZ")
]
qs = cirq.LineQubit.range(5)
logical_ops = get_observables(generators)
logical_qs = cirq.LineQubit.range(len(logical_ops))
logical_op_map = []
for i, (xbar, zbar) in enumerate(logical_ops):
    logical_op_map.append((cirq.X.on(logical_qs[i]), stim_pauli_string_to_cirq(xbar)))
    logical_op_map.append((cirq.Z.on(logical_qs[i]), stim_pauli_string_to_cirq(zbar)))

print("Logical operators")
for logical_op, physical_op in logical_op_map:
    logical_op_stim = cirq_pauli_string_to_stim(logical_op, logical_qs)
    physical_op_stim = cirq_pauli_string_to_stim(physical_op, qs)
    print(logical_op_stim, physical_op_stim)