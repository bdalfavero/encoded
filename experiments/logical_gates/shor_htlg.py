import stim
import htlogicalgates as htlg
from encoded.htlg_interface import stim_pauli_string_to_htlg_str

stabilizers = [
    stim.PauliString("ZZ_______"),
    stim.PauliString("_ZZ______"),
    stim.PauliString("___ZZ____"),
    stim.PauliString("____ZZ___"),
    stim.PauliString("______ZZ_"),
    stim.PauliString("_______ZZ"),
    stim.PauliString("XXXXXX___"),
    stim.PauliString("___XXXXXX")
]
logical_xs = [stim.PauliString("ZZZZZZZZZ")]
logical_zs = [stim.PauliString("XXXXXXXXX")]

stabilizers_htlg = [stim_pauli_string_to_htlg_str(stab) for stab in stabilizers]
logical_zs_htlg = [stim_pauli_string_to_htlg_str(zbar) for zbar in logical_zs]
logical_xs_htlg = [stim_pauli_string_to_htlg_str(xbar) for xbar in logical_xs]

connectivity = htlg.Connectivity("circular", num_qubits=9)
circuit = htlg.Circuit(1)
circuit.h(0)
stab_code = htlg.StabilizerCode(logical_xs_htlg, logical_zs_htlg, stabilizers_htlg)
circ, status = htlg.tailor_logical_gate(stab_code, connectivity, circuit, num_cz_layers=3, log_to_console=True)
print(status)
print(circ)