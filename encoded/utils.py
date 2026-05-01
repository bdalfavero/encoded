from typing import List, Optional, Dict
from functools import reduce
import numpy as np
import stim
import cirq
import openfermion as of

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


def to_groups_of(groups: List[List[cirq.PauliString]]) -> List[of.QubitOperator]:
    """Convert groups from List[List[cirq.PauliString]] to List[of.QubitOperator]."""
    maps = {cirq.X: "X", cirq.Y: "Y", cirq.Z: "Z"}

    groups_of: List[of.QubitOperator] = []
    for group in groups:
        group_of = of.QubitOperator()

        for p in group:
            group_of += of.QubitOperator(" ".join([f"{maps[v]}{k.x}" for k, v in p._qubit_pauli_map.items()]), coefficient=p.coefficient)
        groups_of.append(group_of)

    return groups_of


def cirq_pauli_sum_to_openfermion_qubop(psum: cirq.PauliSum) -> of.QubitOperator:
    groups = [[ps for ps in psum]]
    qubop_list = to_groups_of(groups)
    assert len(qubop_list) == 1
    return qubop_list[0]


def stim_pauli_string_to_circuit(pstring: stim.PauliString) -> stim.Circuit:
    circuit = stim.Circuit()
    for i, p in enumerate(pstring):
        if p == 1:
            circuit.append("X", [i])
        elif p == 2:
            circuit.append("Y", [i])
        elif p == 3:
            circuit.append("Z", [i])
    return circuit


def stim_bits_to_floats(stim_bits: np.ndarray) -> np.ndarray:
    floats = []
    for bit in stim_bits:
        if bit:
            floats.append(-1.)
        else:
            floats.append(1.)
    return np.array(floats)