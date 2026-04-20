from typing import List
import numpy as np

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

    max_j = min(A_copy.shape[0], A_copy.shape[1])
    for j in range(max_j):
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
        if first_i >= min(A_rref.shape):
            first_i -= 1
            break
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


def _pivot_columns(A: np.ndarray) -> List[int]:
    """Find the pivot columns of the binary matrix A in RREF."""

    i = 0 # Index of row where the pivot is.
    j = 0 # Index of column we are currently searching.
    pivot_columns = []
    while i < A.shape[0]:
        if A[i, j]:
            i += 1
            pivot_columns.append(j)
        j += 1
        if j >= A.shape[1]:
            break
    return pivot_columns


def solve_boolean_system(A, b, verbose: bool=False):
    A_rref, b_rref = _boolean_rref(A, b)
    if verbose:
        print("A_rref=\n", A_rref)
        print("b_rref=\n", b_rref)
    x = _boolean_backsub_solve(A_rref, b_rref)
    return x