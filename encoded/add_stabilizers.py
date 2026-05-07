from typing import List, Tuple, Optional, Set, Collection
from warnings import warn
import itertools
import functools
from random import randrange, seed, sample
from copy import deepcopy
import numpy as np
import stim
from encoded.decompose_operators import generators_to_matrix
from encoded.binary_linalg import solve_boolean_system, _boolean_rref, enumerate_all_solutions

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

    # for gen in generators:
    #     print(gen)
    # for err in errors:
    #     print(err)

    generator_matrix = generators_to_matrix(generators)
    error_matrix = generators_to_matrix(errors)
    pstring_matrix = np.hstack((generator_matrix, error_matrix))
    nq = max([len(ps) for ps in generators + errors])
    lamb = metric_tensor(nq)
    A = pstring_matrix.T @ lamb
    b = np.array([False] * len(generators) + [True] * len(errors))
    return A, b


def add_stabilizer(
    generators: List[stim.PauliString], errors: List[stim.PauliString], extra_support: Optional[stim.PauliString]=None,
    verbose: bool=False
) -> List[stim.PauliString]:
    A, b = _form_linear_system(generators, errors)
    if verbose:
        print("A=\n", A)
        print("b=\n", b)
    A_rref, b_rref = _boolean_rref(A, b)
    if verbose:
        print("A_rref=\n", A_rref)
        print("b_rref=\n", b_rref)
    solutions = enumerate_all_solutions(A_rref, b_rref)
    candidate_strings = []
    for x in solutions:
        xs = x[:x.size // 2]
        zs = x[(x.size // 2):]
        pstring = stim.PauliString.from_numpy(xs=xs, zs=zs)
        candidate_strings.append(pstring)
    new_generator = min(candidate_strings, key=lambda ps: ps.weight)
    if extra_support is not None:
        new_generator += extra_support
        # Add identity to all the original generators.
        id_extra = stim.PauliString("I" * len(extra_support))
        old_generators = [gen + id_extra for gen in generators]
    else:
        old_generators = deepcopy(generators)
    for err in errors:
        assert not new_generator.commutes(err), f"[{new_generator}, {err}] = 0"
    new_generators = old_generators + [new_generator]
    return new_generators


def generate_stabilizer_elements(generators: List[stim.PauliString]) -> List[stim.PauliString]:
    nq = max([len(ps) for ps in generators])
    elements = []
    for string in itertools.chain.from_iterable(itertools.combinations(generators, r) for r in range(len(generators) + 1)):
        elements.append(
            functools.reduce(lambda a, b: a * b, string, stim.PauliString('_' * nq))
        )
    return elements


def stim_strings_equal_up_to_phase(ps_a: stim.PauliString, ps_b: stim.PauliString) -> bool:
    return list(ps_a) == list(ps_b)


# TODO Replace this with a function that attempts to solve the system of equations.
def test_group_membership(operator: stim.PauliString, generators: List[stim.PauliString]) -> bool:
    """Test if the operator belongs to the group with the given generators."""

    # If we pass in an operator that is too shot, add identities to the end.
    nq = max([len(gen) for gen in generators])
    if len(operator) < nq:
        new_operator = operator + stim.PauliString("_" * (nq - len(operator)))
    else:
        new_operator = operator

    equality_tests = []
    for ps in generate_stabilizer_elements(generators):
        test = stim_strings_equal_up_to_phase(new_operator, ps)
        equality_tests.append(test)
    return any(equality_tests)


def knill_laflamme_cost_function(generators: List[stim.PauliString], errors: List[float], weights: List[float]) -> float:
    """Cost function from the RL paper."""

    total_loss = 0.
    for weight, err in zip(weights, errors):
        anticommutation_tests = [not gen.commutes(err) for gen in generators]
        in_stabilizer_group = test_group_membership(err, generators)
        if any(anticommutation_tests) or in_stabilizer_group:
            # The error is correctable, so K_mu = 1.
            total_loss -= weight
    return total_loss


def knill_laflamme_correctable_cost_function(generators: List[stim.PauliString], errors: List[float], weights: List[float]) -> float:
    """Check pairs of errors."""

    total_loss = 0.
    for weight1, err1 in zip(weights, errors):
        for weight2, err2 in zip(weights, errors):
            weight = weight1 * weight2
            err = err1 * err2
            anticommutation_tests = [not gen.commutes(err) for gen in generators]
            in_stabilizer_group = test_group_membership(err, generators)
            if any(anticommutation_tests) or in_stabilizer_group:
                # The error is correctable, so K_mu = 1.
                total_loss -= weight
    return total_loss


def get_uncorrectable_errors(generators: List[stim.PauliString], errors: List[stim.PauliString]) -> List[stim.PauliString]:
    """Get the products of errors that the code cannot correct."""

    uncorrectable_errs = []
    for i, e_i in enumerate(errors):
        for j, e_j in enumerate(errors):
            e = e_i * e_j
            anti_commute_tests = [not e.commutes(g) for g in generators]
            in_group = test_group_membership(e, generators)
            if not (any(anti_commute_tests) or in_group):
                uncorrectable_errs.append(e)
    return uncorrectable_errs


def build_code_randomly(
    generators: List[stim.PauliString], errors: List[stim.PauliString], extra_support: Optional[stim.PauliString]=None,
    max_iter: int = 1_000, seed_val: int=137, errors_per_round=1
) -> List[stim.PauliString]:
    """Build a code by randomly picking the error(s) that the new stabilizer will anticommute with."""

    seed(seed_val)
    new_generators = deepcopy(generators)
    uncorrectables = get_uncorrectable_errors(new_generators, errors)
    iters = 0
    while len(uncorrectables) != 0:
        # i = randrange(len(uncorrectables))
        inds = sample(range(len(uncorrectables)), errors_per_round)
        errs = [uncorrectables[i] for i in inds]
        new_generators = add_stabilizer(new_generators, errs, extra_support=extra_support, verbose=False)
        # print("New stabilizer:", new_generators[-1])
        uncorrectables = get_uncorrectable_errors(new_generators, errors)
        # If there are now more qubits in the stabilizers than the erros, add qubits to the errors.
        nq = max(len(gen) for gen in new_generators)
        for i in range(len(uncorrectables)):
            if len(uncorrectables[i]) < nq:
                uncorrectables[i] += stim.PauliString("_" * (nq - len(uncorrectables[i])))
        iters += 1
        if iters > max_iter:
            warn(f"Exceeded max iterations ({max_iter}).")
            break
    return new_generators


def prune_duplicate_pauli_strings(strings: List[stim.PauliString]) -> List[stim.PauliString]:
    """stim.PauliString objects are not hashable, so you can't make a set of them. This function
    removed duplicated from the list."""

    new_strings = []
    for pstring in strings:
        if not any([pstring == ps for ps in new_strings]):
            new_strings.append(pstring)
    return new_strings


def all_new_codes_for_errors(
    generators: Collection[stim.PauliString], errors: List[stim.PauliString],
    extra_support: Optional[stim.PauliString] = None
) -> List[stim.PauliString]:
    """Given a code and a set of errors to correct, enumerate all new stabilizers that we could add."""

    A, b = _form_linear_system(generators, errors)
    A_rref, b_rref = _boolean_rref(A, b)
    solutions = enumerate_all_solutions(A_rref, b_rref)
    candidate_strings = []
    for x in solutions:
        xs = x[:x.size // 2]
        zs = x[(x.size // 2):]
        pstring = stim.PauliString.from_numpy(xs=xs, zs=zs)
        candidate_strings.append(pstring)
    new_codes = []
    for new_generator in candidate_strings:
        if extra_support is not None:
            new_generator += extra_support
            # Add identity to all the original generators.
            id_extra = stim.PauliString("I" * len(extra_support))
            old_generators = [gen + id_extra for gen in generators]
        else:
            old_generators = deepcopy(generators)
        for err in errors:
            assert not new_generator.commutes(err), f"[{new_generator}, {err}] = 0"
        new_generators = old_generators + [new_generator]
        new_codes.append(new_generators)
    return new_codes


def random_depth_first_search(
    stabilizers: List[stim.PauliString], errors: List[stim.PauliString],
    steps: int, max_tries: int, seed_val: int
) -> List[stim.PauliString]:
    """Do a depth-first search, at each step picking a random error to anticommute with
    and a random stabilizer from the list of systems of equations."""

    best_code = deepcopy(stabilizers)
    best_num_uncorredtable = len(get_uncorrectable_errors(stabilizers))

    for _ in range()


if __name__ == "__main__":
    stabilizers = [stim.PauliString("ZZ_"), stim.PauliString("_ZZ")]
    err = stim.PauliString("Z__")
    new_codes = all_new_codes_for_errors(stabilizers, [err], extra_support=stim.PauliString("X"))
    for new_code in new_codes:
        print("New code:")
        for stab in new_code:
            print(stab)