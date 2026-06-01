from typing import List, Tuple, Optional
from copy import deepcopy
from warnings import warn
from itertools import combinations, product
import numpy as np
import stim
from pysat.formula import Atom, Formula, And, Or, XOr, Neg, PYSAT_TRUE, PYSAT_FALSE
from pysat.solvers import Glucose3
from encoded.stabilizer_code import StabilizerCode

class VariableString:
    """A representation of a Pauli string with unknowns.
    Attributes:
    n - The number of qubits.
    idx - The index of this string in the set, e.g. 0 for the first Pauli string we are solving for."""

    def __init__(self, n: int, idx: int):
        self._n = n
        self._idx = idx
        self._x_atoms = []
        self._z_atoms = []
        for i in range(n):
            self._x_atoms.append(Atom(f"x_{idx}_{i}"))
            self._z_atoms.append(Atom(f"z_{idx}_{i}"))
        for atom in self._x_atoms + self._z_atoms:
            atom.clausify()
    
    @property
    def n(self):
        return self._n
    
    @property
    def atoms(self):
        return (self._x_atoms, self._z_atoms)
    
    def assign_from_model(self, model: List[int]) -> Tuple[np.array, np.array]:
        """Given a satisfying assignment from pysat, get the x and z components of the string
        as binary arrays. e.g. if the variable string has atoms with names ([1, 2, 3], [4, 5, 6])
        and the assignment is [-1, 2, 3, -4, -5, -6, 10, -11], then return ([False, True, True], [False, False, False])."""

        x_names = [x_atom.name for x_atom in self._x_atoms]
        z_names = [z_atom.name for z_atom in self._z_atoms]
        x_bools = [None] * len(x_names)
        z_bools = [None] * len(z_names)
        for assignment in model:
            if -assignment in x_names:
                i = x_names.index(-assignment)
                x_bools[i] = False
            if assignment in x_names:
                i = x_names.index(assignment)
                x_bools[i] = True
            if -assignment in z_names:
                i = z_names.index(-assignment)
                z_bools[i] = False
            if assignment in z_names:
                i = z_names.index(assignment)
                z_bools[i] = True
        for i, x_bool in enumerate(x_bools):
            if x_bool is None:
                raise ValueError(f"Value at position {i} of x_bools is None.")
        for i, z_bool in enumerate(z_bools):
            if z_bool is None:
                raise ValueError(f"Value at position {i} of z_bools is None.")
        return (np.array(x_bools), np.array(z_bools))
    
    def assign_to_stim(self, model: List[int]) -> stim.PauliString:
        xs, zs = self.assign_from_model(model)
        return stim.PauliString.from_numpy(xs=xs, zs=zs)
    
    def assignment_from_stim(self, pstring: stim.PauliString) -> List[int]:
        """Given a stim.PauliString, return an assignment of the variables."""

        assert len(pstring) == self.n
        xs, zs = pstring.to_numpy()
        assignment = []
        for i, x in enumerate(xs):
            if x:
                assignment.append(self._x_atoms[i])
            else:
                assignment.append(Neg(self._x_atoms[i]))
        for i, z in enumerate(zs):
            if z:
                assignment.append(self._z_atoms[i])
            else:
                assignment.append(Neg(self._z_atoms[i]))
        return assignment

    def non_weight_zero_constraint(self):
        """Build a formula that asserts that this string cannot be the identity."""

        return Or(*self._x_atoms, *self._z_atoms)
    

def fixed_to_variable_commutation_constraint(row: np.ndarray, var_string: VariableString, commutes: bool) -> Formula:
    """Given a Pauli string as a row of a check matrix and a variable string,
    build a formula that tests wether the variable string would (anti-)commute with the
    fixed row."""

    n = row.size // 2
    xs, zs = row[:n], row[n:]
    x_vars, z_vars = var_string.atoms
    # If we have a row [rx, rz] and a variable string [x, z],
    # the contraints become (x_i1 ^ ... ^ x_iq) ^ (z_j1 ^ ... ^ z_js),
    # where i1, ..., iq are the indices for which rz[i] = 1
    # and j1, ..., js are the indices for which rx[j] = 1.
    x_ands = [z_vars[i] for i in range(n) if xs[i]]
    z_ands = [x_vars[i] for i in range(n) if zs[i]]
    if len(x_ands) == 1 and len(z_ands) == 0:
        if commutes:
            return Neg(x_ands[0])
        else:
            return x_ands[0]
    elif len(x_ands) == 0 and len(z_ands) == 1:
        if commutes:
            return Neg(z_ands[0])
        else:
            return z_ands[0]
    else:
        if commutes:
            return Neg(XOr(*x_ands, *z_ands))
        else:
            return XOr(*x_ands, *z_ands)


class CommutationConstraint:
    """A constraints that says the new stabilizers should (anti-)commute with some set of Pauli strings."""

    def __init__(self, strings: StabilizerCode, commutes: bool=True, all_commute: bool=True):
        self._strings = strings
        self._commutes = commutes
        self._all_commute = all_commute
    
    @property
    def n(self):
        return self._strings.n
    
    def to_formula(self, var_string: VariableString) -> Formula:
        """Convert this constraint to a pysat Formula."""

        assert self.n == var_string.n

        subformulas = []
        for row in self._strings.check_matrix:
            subformulas.append(fixed_to_variable_commutation_constraint(row, var_string, self._commutes))
        if self._all_commute:
            return And(*subformulas)
        else:
            return Or(*subformulas)


class ErrorConstraint:
    """This constraint says that at least one the variable strings should anticommute with the
    given error. There should be one of these contraints per error we wish to correct."""

    def __init__(self, err: stim.PauliString, code: Optional[StabilizerCode]=None):
        self._err = err
        self._code = code
        if self._code is not None:
            self._code_anticommutators = [not err.commutes(stabilizer) for stabilizer in self._code.to_stim()]
        else:
            self._code_anticommutators = None
    
    def to_formula(self, var_strings: List[VariableString]) -> Formula:
        xs, zs = self._err.to_numpy()
        row = np.hstack((xs, zs))
        subformulas = []
        for var_string in var_strings:
            subformulas.append(fixed_to_variable_commutation_constraint(row, var_string, False))
        if self._code is None:
            return Or(*subformulas)
        else:
            anticommutator_atoms = []
            for bool in self._code_anticommutators:
                if bool:
                    anticommutator_atoms.append(PYSAT_TRUE)
                else:
                    anticommutator_atoms.append(PYSAT_FALSE)
            return Or(*anticommutator_atoms, *subformulas)


def solve_single_stabilizer(code: StabilizerCode, err: stim.PauliString, pad: bool=False, new_support: str="Z") -> stim.PauliString:
    """Find a new stabilizer so that the code can correct a new error of our choice."""

    if pad:
        code.pad(1)
    n = code.n
    error_strings = StabilizerCode.from_stim([err])
    if pad:
        error_strings.pad(1)
    assert error_strings.n == code.n
    var_string = VariableString(n, 0)
    nwz_formula = var_string.non_weight_zero_constraint()
    code_commute_constraint = CommutationConstraint(code, True)
    # error_anticommute_constraint = CommutationConstraint(error_strings, False, all_commute=False)
    error_anticommute_constraint = ErrorConstraint(err, code=code)
    code_commute_formula = code_commute_constraint.to_formula(var_string)
    error_formula = error_anticommute_constraint.to_formula([var_string])
    total_formula = And(code_commute_formula, error_formula, nwz_formula)
    total_formula.clausify()
    g = Glucose3()
    for clause in list(total_formula):
        g.add_clause(clause)
    if not g.solve():
        warn("Solve did not succeed.")
    model = g.get_model()
    new_pstring = var_string.assign_to_stim(model)
    # Modify the Pauli on the last (added) qubit if needed.
    mask = list(new_pstring)
    new_mask = deepcopy(mask)
    if pad:
        if new_support == "Z":
            new_mask[-1] = 3
            new_pstring = stim.PauliString(new_mask)
        elif new_support == "X":
            new_mask[-1] = 1
            new_pstring = stim.PauliString(new_mask)
        else:
            raise ValueError(f"new_support must be X or Z, but got {new_support}")
    return new_pstring


def enumerate_errors(n: int, w: int) -> List[stim.PauliString]:
    """Enumerate all errors on n qubits with Pauli weight w."""

    errs = []
    for support in combinations(range(n), w):
        for paulis in product(range(1, 4), repeat=w):
            mask = [0] * n
            for index, pauli in zip(support, paulis):
                mask[index] = pauli
            errs.append(stim.PauliString(mask))
    return errs


def increase_distance(code: StabilizerCode, d: int, m: int) -> List[stim.PauliString]:
    """Add m new stabilizers on m new qubits to increase the distance of a code by 1."""

    n_prime = code.n + m
    code.pad(m)
    errs = enumerate_errors(n_prime, d)
    var_strings = [VariableString(n_prime, i) for i in range(m)]
    nwz_formulas = [var_string.non_weight_zero_constraint() for var_string in var_strings]
    code_commute_constraint = CommutationConstraint(code, True)
    error_anticommute_constraints = [ErrorConstraint(err) for err in errs]
    code_commute_formulas = [code_commute_constraint.to_formula(var_string) for var_string in var_strings]
    error_formulas = [constraint.to_formula(var_strings) for constraint in error_anticommute_constraints]
    total_formula = And(*code_commute_formulas, *error_formulas, *nwz_formulas)
    total_formula.clausify()
    g = Glucose3()
    for clause in list(total_formula):
        g.add_clause(clause)
    if g.solve():
        model = g.get_model()
        new_pstrings = [var_string.assign_to_stim(model) for var_string in var_strings]
        return new_pstrings
    else:
        return None


if __name__ == "__main__":
    all_errs = enumerate_errors(4, 2)
    for err in all_errs:
        print(err)