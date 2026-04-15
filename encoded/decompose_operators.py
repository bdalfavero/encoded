from typing import List, Tuple
from math import prod
import numpy as np
from scipy.optimize import milp, LinearConstraint
import stim

def _swap_row(arr: np.ndarray, i: int, j: int):
    temp = arr[i, :].copy()
    arr_copy = arr.copy()
    arr_copy[i, :] = arr_copy[j, :]
    arr_copy[j, :] = temp
    return arr_copy


def _swap_elems(b: np.ndarray, i, j):
    b_copy = b.copy()
    temp = b[i]
    b_copy[i] = b[j]
    b_copy[j] = temp
    return b_copy


def _boolean_rref(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    # assert A.shape[1] <= A.shape[0]

    A_copy = A.copy()
    b_copy = b.copy()

    for j in range(A.shape[1]):
        # Find the first index i s.t. A[i, j] = 1.
        found = False
        for idx_first_one in range(j, A_copy.shape[0]):
            if A_copy[idx_first_one, j]:
                found = True
                break
        # Swap that row with the j^th row.
        A_copy = _swap_row(A_copy, idx_first_one, j)
        b_copy = _swap_elems(b_copy, idx_first_one, j)
        # Eliminate all other rows i where A[i, j] = 1.
        for i in range(j+1, A.shape[0]):
            if A_copy[i, j]:
                A_copy[i, :] = A_copy[i, :] ^ A_copy[j, :]
                b_copy[i] = b_copy[i] ^ b_copy[j]
    return A_copy, b_copy


def _boolean_backsub_solve(A_rref: np.ndarray, b_rref: np.ndarray) -> np.ndarray:
    x = np.zeros(A_rref.shape[1]).astype(bool)
    # Find the first value of i s.t. A[i, i] != 1.
    found = False
    for first_i in range(A_rref.shape[1]):
        if not A_rref[first_i, first_i]:
            found = True
            break
    if not found:
        first_i += 1
    for i in range(first_i - 1, -1, -1):
        x_i = False
        for j in range(i+1, first_i):
            x_i ^= A_rref[i, j] and x[j]
        x_i ^= b_rref[i]
        x[i] = x_i
    return x


def solve_boolean_system(A, b):
    A_rref, b_rref = _boolean_rref(A, b)
    x = _boolean_backsub_solve(A_rref, b_rref)
    return x


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
