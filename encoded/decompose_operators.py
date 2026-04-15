from typing import List
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
    b_copy[j] = b[i]
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
                A_copy[i, :] = A_copy[i, :] ^ A_copy[idx_first_one, :]
                b_copy[i] = b[i] ^ b_copy[idx_first_one]
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
            x_i ^= A_rref[j, i] and x[j]
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
