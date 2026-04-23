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