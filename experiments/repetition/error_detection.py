import stim
import htlogicalgates as htlg
from encoded.utils import get_observables
from encoded.prep_circuit import cb_prep_circuit
from encoded.htlg_interface import stim_pauli_string_to_htlg_str, htlg_circuit_to_stim

stabilizers = [
    stim.PauliString("ZZ_"),
    stim.PauliString("_ZZ")
]
logical_ops = get_observables(stabilizers)
logical_zs = [t[1] for t in logical_ops]
logical_xs = [t[0] for t in logical_ops]
print("Logical Z:")
for zbar in logical_zs:
    print(zbar)
print("Logical X:")
for xbar in logical_xs:
    print(xbar)

stabs_htlg = [stim_pauli_string_to_htlg_str(stab) for stab in stabilizers]
logical_zs_htlg = [stim_pauli_string_to_htlg_str(zbar) for zbar in logical_zs]
logical_xs_htlg = [stim_pauli_string_to_htlg_str(xbar) for xbar in logical_xs]

encoding_ckt = cb_prep_circuit(stabilizers, logical_zs, [False])
print("Encoding circuit")
print(encoding_ckt)

# Get the physical version of the logical circuit.
logical_gate = htlg.Circuit(1)
logical_gate.x(0)
stab_code = htlg.StabilizerCode(logical_xs_htlg, logical_zs_htlg, stabs_htlg)
connectivity = htlg.Connectivity("circular", num_qubits=3)
logical_circ, status = htlg.tailor_logical_gate(stab_code, connectivity, logical_gate, num_cz_layers=2)
logical_ckt_stim = htlg_circuit_to_stim(logical_circ)
print("Logical circuit")
print(logical_ckt_stim)

total_ckt = encoding_ckt + logical_ckt_stim
for stab in stabilizers:
    total_ckt.append("MPP", stab)
total_ckt.append("MPP", logical_zs[0])
sampler = total_ckt.compile_sampler()
result = sampler.sample(10)
print(result)