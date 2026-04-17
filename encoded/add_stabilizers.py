from typing import List, Tuple
import numpy as np
import stim
from encoded.decompose_operators import generators_to_matrix
from encoded.binary_linalg import solve_boolean_system

def metric_tensor(nq: int) -> np.ndarray:
    id_nq = np.eye(nq).astype(bool)
    zeros_nq = np.zeros((nq, nq)).astype(bool)
    return np.vstack((
        np.hstack((zeros_nq, id_nq)),
        np.hstack((id_nq, zeros_nq))
    ))


def _form_linear_system(generators: List[stim.PauliString], errors: List[stim.PauliString]) -> Tuple[np.ndarray, np.ndarray]:
    """Form the linear system that requires the new generator to commute with the existing
    generators and anticommute with the given errors."""

    generator_matrix = generators_to_matrix(generators)
    error_matrix = generators_to_matrix(errors)
    pstring_matrix = np.hstack((generator_matrix, error_matrix))
    nq = max([len(ps) for ps in generators + errors])
    lamb = metric_tensor(nq)
    A = pstring_matrix.T @ lamb
    b = np.array([False] * len(generators) + [True] * len(errors))
    return A, b


def add_stabilizer(generators: List[stim.PauliString], errors: List[stim.PauliString]) -> stim.PauliString:
    A, b = _form_linear_system(generators, errors)
    x = solve_boolean_system(A, b)
    xs = x[:x.size // 2]
    zs = x[(x.size // 2):]
    return stim.PauliString.from_numpy(xs, zs)