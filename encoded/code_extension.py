import stim

def encoding_unitary_for_new_stabilizer(stabilizer: stim.PauliString) -> stim.Circuit:
    """Is we are adding the stabilizer generator g to a code, we need to add on to the
    encoding circuit. This implementation follows algorithm II of the paper. We assume the last qubit
    of the Pauli string is the new one."""

    pre_cnot_ckt = stim.Circuit()
    cnot_ckt = stim.Circuit()
    post_cnot_ckt = stim.Circuit()
    for i, p in enumerate(stabilizer):
        if p == 1: # Pauli X.
            if i != len(stabilizer) - 1:
                pre_cnot_ckt.append("H", i)
            post_cnot_ckt.append("H", i)
        if p == 2:
            if i != len(stabilizer) - 1:
                pre_cnot_ckt.append("H", i)
                pre_cnot_ckt.append("S", i)
            post_cnot_ckt.append("H", i)
            post_cnot_ckt.append("S", i)
        if p in [1, 2, 3] and i != len(stabilizer) - 1:
            cnot_ckt.append("CNOT", [i, len(stabilizer)-1])
    return pre_cnot_ckt + cnot_ckt + post_cnot_ckt