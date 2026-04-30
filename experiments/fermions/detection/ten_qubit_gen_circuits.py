"""Generate logical circuits for the [[10, 6]] Fermi-Hubbard code."""

import pickle
import stim
import htlogicalgates as htlg
from encoded.utils import get_observables
from encoded.prep_circuit import cb_prep_circuit
from encoded.htlg_interface import stim_pauli_string_to_htlg_str, htlg_circuit_to_stim

nq = 10
stabilizers = [
    stim.PauliString("ZIZIZIZIII"),
    stim.PauliString("IZIZIZIZII"),
    stim.PauliString("ZZIIZZIIZI"),
    stim.PauliString("ZIZIIZIZIZ")
]
logical_ops = get_observables(stabilizers)
logical_zs = [t[1] for t in logical_ops]
logical_xs = [t[0] for t in logical_ops]

stabs_htlg = [stim_pauli_string_to_htlg_str(stab) for stab in stabilizers]
logical_zs_htlg = [stim_pauli_string_to_htlg_str(zbar) for zbar in logical_zs]
logical_xs_htlg = [stim_pauli_string_to_htlg_str(xbar) for xbar in logical_xs]

encoding_ckt = cb_prep_circuit(stabilizers, logical_zs, [False] * 6)
print("Encoding circuit")
print(encoding_ckt)

# Get the circuit for Hadamard
logical_gate = htlg.Circuit(6)
logical_gate.h(0)
stab_code = htlg.StabilizerCode(logical_xs_htlg, logical_zs_htlg, stabs_htlg)
connectivity = htlg.Connectivity("circular", num_qubits=nq)
logical_circ, status = htlg.tailor_logical_gate(
    stab_code, connectivity, logical_gate, num_cz_layers=6, time_limit=2 * 3.6e3, log_to_console=True,
    optimize=False
)
print(status)
logical_ckt_stim1 = htlg_circuit_to_stim(logical_circ)
print("Logical circuit")
print(logical_ckt_stim1)

# Repeat with CNOT.
logical_gate = htlg.Circuit(6)
logical_gate.cx(0, 1)
stab_code = htlg.StabilizerCode(logical_xs_htlg, logical_zs_htlg, stabs_htlg)
connectivity = htlg.Connectivity("circular", num_qubits=nq)
logical_circ, status = htlg.tailor_logical_gate(stab_code, connectivity, logical_gate, num_cz_layers=6)
print(status)
logical_ckt_stim2 = htlg_circuit_to_stim(logical_circ)
print("Logical circuit")
print(logical_ckt_stim2)

output_dict = {
    "encoding": encoding_ckt,
    "logical_hadamard": logical_ckt_stim1,
    "logical_cnot": logical_ckt_stim2
}
with open("ten_qubit_circuits.pkl", "wb") as f:
    pickle.dump(output_dict, f)