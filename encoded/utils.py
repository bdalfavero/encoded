from typing import List, Optional, Dict
from functools import reduce
import stim
import cirq

def stim_pauli_string_to_cirq(stim_pauli: stim.PauliString) -> cirq.PauliString:
    """Convert a stim PauliString to a cirq PauliString."""

    qs = cirq.LineQubit.range(len(stim_pauli))
    cirq_ops = []
    for i, p in enumerate(stim_pauli):
        if p == 1:
            cirq_ops.append(cirq.X.on(qs[i]))
        elif p == 2:
            cirq_ops.append(cirq.Y.on(qs[i]))
        elif p == 3:
            cirq_ops.append(cirq.Z.on(qs[i]))
        else:
            pass
    return stim_pauli.sign * reduce(lambda a, b: a * b, cirq_ops, cirq.PauliString())


def cirq_pauli_string_to_stim(cirq_pauli: cirq.PauliString, qs: Optional[List[cirq.Qid]]=None) -> stim.PauliString:
    """Convert a cirq PauliString to stim. Note that the coefficient will be automatically
    converted to one of [1, -1, i, -i] based on whichever is closest."""

    if qs == None:
        qs = list(cirq_pauli.qubits)
    assert set(cirq_pauli.qubits).issubset(set(qs))

    dense_pstring = cirq_pauli.dense(qs)
    pauli_ints = list(dense_pstring.pauli_mask)
    coeff = dense_pstring.coefficient
    int_char_conversion = {
        0: 'I',
        1: 'X',
        2: 'Y',
        3: 'Z'
    }
    pauli_chars = [int_char_conversion[p] for p in pauli_ints]
    if abs(coeff.real) >= abs(coeff.imag):
        if coeff.real >= 0:
            sign_str = ''
        else:
            sign_str = '-'
    else:
        if coeff.imag >= 0:
            sign_str = 'i'
        else:
            sign_str = '-i'
    return stim.PauliString(sign_str + ''.join(pauli_chars))

def get_observables(
    stabilizers: list[stim.PauliString],
) -> list[tuple[stim.PauliString, stim.PauliString]]:
    """See
    https://quantumcomputing.stackexchange.com/questions/37812/how-to-find-a-set-of-independent-logical-operators-for-a-stabilizer-code-with-st"""
    
    completed_tableau = stim.Tableau.from_stabilizers(
        stabilizers,
        allow_redundant=True,
        allow_underconstrained=True,
    )

    observables = []
    for k in range(len(completed_tableau))[::-1]:
        z = completed_tableau.z_output(k)
        if z in stabilizers:
            break
        x = completed_tableau.x_output(k)
        observables.append((x, z))

    return observables