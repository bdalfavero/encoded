import pickle
import numpy as np
import matplotlib.pyplot as plt
import stim
import htlogicalgates as htlg
from encoded.utils import get_observables
from encoded.prep_circuit import cb_prep_circuit
from encoded.htlg_interface import stim_pauli_string_to_htlg_str, htlg_circuit_to_stim

stabilizers = [
    stim.PauliString("Z_Z_"),
    stim.PauliString("_Z_Z")
]
logical_ops = get_observables(stabilizers)
logical_zs = [t[1] for t in logical_ops]
logical_xs = [t[0] for t in logical_ops]

stabs_htlg = [stim_pauli_string_to_htlg_str(stab) for stab in stabilizers]
logical_zs_htlg = [stim_pauli_string_to_htlg_str(zbar) for zbar in logical_zs]
logical_xs_htlg = [stim_pauli_string_to_htlg_str(xbar) for xbar in logical_xs]

encoding_ckt = cb_prep_circuit(stabilizers, logical_zs, [False] * 2)
print("Encoding circuit")
print(encoding_ckt)

# Get the physical version of the logical circuit.
logical_gate = htlg.Circuit(2)
logical_gate.h(0)
# logical_gate.cx(0, 1)
stab_code = htlg.StabilizerCode(logical_xs_htlg, logical_zs_htlg, stabs_htlg)
connectivity = htlg.Connectivity("circular", num_qubits=4)
logical_circ, status = htlg.tailor_logical_gate(
    stab_code, connectivity, logical_gate, num_cz_layers=4,
    time_limit=3.6e3, log_to_console=True, optimize=True
)
print(status)
logical_ckt_stim = htlg_circuit_to_stim(logical_circ)
print(logical_circ.to_qiskit())
print(logical_circ)
print("Logical circuit")
print(logical_ckt_stim)

# Get the physical version of the logical circuit.
logical_gate = htlg.Circuit(2)
logical_gate.cx(0, 1)
logical_circ2, status = htlg.tailor_logical_gate(
    stab_code, connectivity, logical_gate, num_cz_layers=4,
    time_limit=3.6e3, log_to_console=True, optimize=True
)
print(status)
logical_ckt_stim2 = htlg_circuit_to_stim(logical_circ2)
print("Logical circuit")
print(logical_ckt_stim2)

output_dict = {
    "encoding": encoding_ckt,
    "logical_h": logical_ckt_stim,
    "logical_cx": logical_ckt_stim2
}

with open("four_qubit_circuits.pkl", "wb") as f:
    pickle.dump(output_dict, f)

print(logical_xs[0].after(logical_ckt_stim))