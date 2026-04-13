import stim
import cirq
import openfermion as of
from encoded.utils import get_observables, stim_pauli_string_to_cirq

generators = [
    stim.PauliString("ZIZI"),
    stim.PauliString("IZIZ")
]
logical_ops = get_observables(generators)
logical_qs = cirq.LineQubit.range(len(logical_ops))
logical_op_map = {}
for i, (xbar, zbar) in enumerate(logical_ops):
    logical_op_map[cirq.X.on(logical_qs[i])] = stim_pauli_string_to_cirq(xbar)
    logical_op_map[cirq.Z.on(logical_qs[i])] = stim_pauli_string_to_cirq(zbar)

print("Logical operators")
for logical_op, physical_op in logical_op_map.items():
    print(logical_op, physical_op)

ham_fermi = of.hamiltonians.fermi_hubbard(2, 1, 0.1, 1.0)
ham_qubop = of.transforms.jordan_wigner(ham_fermi)
ham_cirq = of.transforms.qubit_operator_to_pauli_sum(ham_qubop)
print("Physical Hamiltonian")
for ps in ham_cirq:
    print(ps)