"""Interface between stim and the htlogicalgates library."""

import stim
import htlogicalgates as htlg

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


def htlg_circuit_to_stim(circuit: htlg.Circuit) -> stim.Circuit:
    """Convert a circuit from htlg to stim."""

    stim_ckt = stim.Circuit()
    for gate, qs in circuit._gates:
        if gate == htlg.circuit.Operation.H:
            stim_ckt.append("H", qs)
        elif gate == htlg.circuit.Operation.S:
            stim_ckt.append("S", qs)
        elif gate == htlg.circuit.Operation.X:
            stim_ckt.append("X", qs)
        elif gate == htlg.circuit.Operation.Y:
            stim_ckt.append("Y", qs)
        elif gate == htlg.circuit.Operation.Z:
            stim_ckt.append("Z", qs)
        elif gate == htlg.circuit.Operation.I:
            stim_ckt.append("I", qs)
        elif gate == htlg.circuit.Operation.SDG:
            stim_ckt.append("S_DAG", qs)
        elif gate == htlg.circuit.Operation.SXDG:
            stim_ckt.append("SQRT_X_DAG", qs)
        elif gate == htlg.circuit.Operation.C_XYZ:
            stim_ckt.append("H", qs)
            stim_ckt.append("S_DAG", qs)
        elif gate == htlg.circuit.Operation.C_ZYX:
            stim_ckt.append("S", qs)
            stim_ckt.append("H", qs)
        elif gate == htlg.circuit.Operation.CX:
            stim_ckt.append("CNOT", list(qs))
        elif gate == htlg.circuit.Operation.CZ:
            stim_ckt.append("CZ", list(qs))
        elif gate == htlg.circuit.Operation.SWAP:
            stim_ckt.append("SWAP", list(qs))
        elif gate == htlg.circuit.Operation.BARRIER:
            continue
        else:
            raise ValueError(f"Unrecoginzed gate {gate}.")
    return stim_ckt