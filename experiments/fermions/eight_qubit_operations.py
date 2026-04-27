import stim
import htlogicalgates as htlg
from encoded.utils import get_observables
from encoded.prep_circuit import cb_prep_circuit

def stim_pauli_string_to_htlg_str(ps: stim.PauliString) -> str:
    ps_str = ""
    for i, p in enumerate(ps):
        if p == 1:
            ps_str += f"X{i} "
        elif p == 2:
            ps_str += f"Y{i} "
        elif p == 3:
            ps_str += f"Z{i} "
        else:
            continue
    return ps_str[:-1]

stabilizers = [
    stim.PauliString("Z_Z_Z_Z_"),
    stim.PauliString("_Z_Z_Z_Z")
]
stabs_htlg = [stim_pauli_string_to_htlg_str(stab) for stab in stabilizers]

logical_ops = get_observables(stabilizers)
logical_zs = [t[1] for t in logical_ops]
logical_xs = [t[0] for t in logical_ops]
logical_zs_htlg = [stim_pauli_string_to_htlg_str(zbar) for zbar in logical_zs]
logical_xs_htlg = [stim_pauli_string_to_htlg_str(xbar) for xbar in logical_xs]
