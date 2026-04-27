import pickle
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import stim
import cirq
import openfermion as of
from encoded.utils import (
    get_observables, stim_pauli_string_to_cirq,
    cirq_pauli_string_to_stim, cirq_pauli_sum_to_openfermion_qubop
)
from encoded.decompose_operators import decompose_pauli_to_logical_operators

generators = [
    stim.PauliString("ZIZIZIZI"),
    stim.PauliString("IZIZIZIZ")
]
qs = cirq.LineQubit.range(8)
logical_ops = get_observables(generators)
logical_qs = cirq.LineQubit.range(len(logical_ops))
logical_op_map = []
for i, (xbar, zbar) in enumerate(logical_ops):
    logical_op_map.append((cirq.X.on(logical_qs[i]), stim_pauli_string_to_cirq(xbar)))
    logical_op_map.append((cirq.Z.on(logical_qs[i]), stim_pauli_string_to_cirq(zbar)))

print("Logical operators")
for logical_op, physical_op in logical_op_map:
    print(logical_op, physical_op)

logical_map_stim = []
for logical_op, physical_op in logical_op_map:
    logical_op_stim = cirq_pauli_string_to_stim(logical_op, logical_qs)
    phsyical_op_stim = cirq_pauli_string_to_stim(physical_op, qs)
    logical_map_stim.append((logical_op_stim, phsyical_op_stim))

print("Mapping")
for lop, pop in logical_map_stim:
    print(lop, pop)

ham_fermi = of.hamiltonians.fermi_hubbard(2, 2, 0.1, 1.0)
ham_qubop = of.transforms.jordan_wigner(ham_fermi)
ham_cirq = of.transforms.qubit_operator_to_pauli_sum(ham_qubop)
# print("Physical Hamiltonian")
# for ps in ham_cirq:
#     print(ps)

print("Convert to logical operators")
logical_strings = []
for ps in ham_cirq:
    stim_ps = cirq_pauli_string_to_stim(ps, qs)
    logical_ps_stim = decompose_pauli_to_logical_operators(stim_ps, logical_map_stim, generators)
    coeff = ps.coefficient / stim_ps.sign
    logical_strings.append(coeff * stim_pauli_string_to_cirq(logical_ps_stim))
    print(ps, "->", logical_strings[-1])
logical_hamiltonian = cirq.PauliSum.from_pauli_strings(logical_strings)

# Look at the spectra of the physical and logical Hamiltonians
print("Physical eigenvalues")
phys_ham_matrix = ham_cirq.matrix(qs)
phys_eigvals, phys_eigvecs = eigh(phys_ham_matrix)
print(phys_eigvals)
print("Logical eigenvalues")
logical_ham_matrix = logical_hamiltonian.matrix(logical_qs)
logical_eigvals, logical_eigvecs = eigh(logical_ham_matrix)
print(logical_eigvals)
# for i, energy in enumerate(logical_eigvals):
#     print(energy, logical_eigvecs[:, i])

# fig, ax = plt.subplots()
# ax.hlines(phys_eigvals, 1., 2., colors=["blue"], label="Physical")
# ax.hlines(logical_eigvals, 3., 4., colors=["tab:orange"], label="Logical")
# ax.set_ylabel("Energy")
# ax.set_xticks([])
# ax.legend(loc="center right")
# plt.show()

ham_of = cirq_pauli_sum_to_openfermion_qubop(logical_hamiltonian)

with open("eight_qubit_logical_hamiltonian.pkl", "wb") as f:
    pickle.dump(ham_of, f)