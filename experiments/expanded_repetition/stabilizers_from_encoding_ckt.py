"""Conjugate the initial stabilizers Z_{k+1}, ..., Z_n through the circuit."""

import stim

ckt = stim.Circuit()
ckt.append("CNOT", [0, 1])
ckt.append("CNOT", [0, 2])

logical_ops_before = [
    stim.PauliString("X__"),
    stim.PauliString("Z__"),
]
stabilizers_before = [
    stim.PauliString("_Z_"),
    stim.PauliString("__Z")
]

logical_ops_after = [lop.after(ckt) for lop in logical_ops_before]
stabilizers_after = [stab.after(ckt) for stab in stabilizers_before]

print("Logical operators:")
for lop in logical_ops_after:
    print(lop)
print("Stabilizers:")
for stab in stabilizers_after:
    print(stab)

# Do a five-qubit code.
# See arXiv:1509.01239
ckt = stim.Circuit()
ckt.append("H", range(1, 5))
ckt.append("CNOT", [4, 0, 3, 0, 2, 0, 1, 0])
ckt.append("CZ", [4, 3, 3, 2, 2, 1, 1, 0, 0, 4])

logical_ops_before = [
    stim.PauliString("X____"),
    stim.PauliString("Z____"),
]
stabilizers_before = [
    stim.PauliString("_Z___"),
    stim.PauliString("__Z__"),
    stim.PauliString("___Z_"),
    stim.PauliString("____Z")
]

logical_ops_after = [lop.after(ckt) for lop in logical_ops_before]
stabilizers_after = [stab.after(ckt) for stab in stabilizers_before]

print("Logical operators:")
for lop in logical_ops_after:
    print(lop)
print("Stabilizers:")
for stab in stabilizers_after:
    print(stab)