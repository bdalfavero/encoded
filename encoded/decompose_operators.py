from typing import List, Tuple
from math import prod
import numpy as np
import stim
from encoded.binary_linalg import solve_boolean_system

def generators_to_matrix(generators: List[stim.PauliString]) -> np.ndarray:
    rows = []
    for ps in generators:
        x, z = ps.to_numpy()
        rows.append(np.hstack((x, z)))
    return np.vstack(rows).T


def decompose_operator_to_product(pstring: stim.PauliString, generators: List[stim.PauliString]) -> List[stim.PauliString]:
    """Given a Pauli string P and set of operators {g1, ..., gn}, find a product of
    the gi operators that equals P."""

    A = generators_to_matrix(generators)
    b_x, b_z = pstring.to_numpy()
    b = np.hstack((b_x, b_z))
    x = solve_boolean_system(A, b)
    pstring_generators = []
    for i, xi in enumerate(x):
        if xi:
            pstring_generators.append(generators[i])
    return pstring_generators


# TODO stim.PauliString is unhashable, but this would be easier with dictionaries.
def decompose_pauli_to_logical_operators(
    pstring: stim.PauliString, logical_op_map: List[Tuple[stim.PauliString, stim.PauliString]], stabilizers: List[stim.PauliString]
) -> stim.PauliString:
    """Convert a Pauli string acting on the physical qubits of a code into a logical Pauli operator.
    
    Arguments:
    pstring - The operator acting on the physical qubits of the code.
    logical_op_map - List of tuples (logical operator, physical operator).
    stabilizers - The stabilizer generators of the code."""

    all_generators = [t[1] for t in logical_op_map] + stabilizers
    generators_of_ps = decompose_operator_to_product(pstring, all_generators)
    logical_components = []
    for gen in generators_of_ps:
        for logical_op, physical_op in logical_op_map:
            if physical_op == gen:
                logical_components.append(logical_op)
    nq_logical = max([len(t[0]) for t in logical_op_map])
    nq_physical = len(pstring)
    logical_id = stim.PauliString("+" + "_" * nq_logical)
    physical_id = stim.PauliString("+" + "_" * nq_physical)
    logical_ps = prod(logical_components, start=logical_id)
    component_product = prod(generators_of_ps, start=physical_id)
    return logical_ps * pstring.sign / component_product.sign
