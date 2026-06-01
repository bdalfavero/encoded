from typing import List, Tuple
from warnings import warn
import numpy as np
import stim
from pysat.formula import Atom, Formula, And, Or, XOr, Neg
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

    def non_weight_zero_constraint(self):
        """Build a formula that asserts that this string cannot be the identity."""

        return Or(*self._x_atoms, *self._z_atoms)


class CommutationConstraint:
    """A constraints that says the new stabilizers should (anti-)commute with some set of Pauli strings."""

    def __init__(self, strings: StabilizerCode, commutes: bool=True):
        self._strings = strings
        self._commutes = commutes
    
    @property
    def n(self):
        return self._strings.n
    
    def to_formula(self, var_string: VariableString) -> Formula:
        """Convert this constraint to a pysat Formula."""

        assert self.n == var_string.n

        x_vars, z_vars = var_string.atoms
        subformulas = []
        for row in self._strings.check_matrix:
            xs, zs = row[:self.n], row[self.n:]
            # If we have a row [rx, rz] and a variable string [x, z],
            # the contraints become (x_i1 ^ ... ^ x_iq) ^ (z_j1 ^ ... ^ z_js),
            # where i1, ..., iq are the indices for which rz[i] = 1
            # and j1, ..., js are the indices for which rx[j] = 1.
            x_ands = [z_vars[i] for i in range(self.n) if xs[i]]
            z_ands = [x_vars[i] for i in range(self.n) if zs[i]]
            if len(x_ands) == 1 and len(z_ands) == 0:
                if self._commutes:
                    subformulas.append(Neg(*x_ands))
                else:
                    subformulas.append(*x_ands)
            elif len(x_ands) == 0 and len(z_ands) == 1:
                if self._commutes:
                    subformulas.append(Neg(*z_ands))
                else:
                    subformulas.append(*z_ands)
            else:
                if self._commutes:
                    subformulas.append(Neg(XOr(*x_ands, *z_ands)))
                else:
                    subformulas.append(XOr(*x_ands, *z_ands))
        return And(*subformulas)


def solve_single_stabilizer(code: StabilizerCode, err: stim.PauliString, pad: bool=False) -> stim.PauliString:
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
    error_anticommute_constraint = CommutationConstraint(error_strings, False)
    code_commute_formula = code_commute_constraint.to_formula(var_string)
    error_formula = error_anticommute_constraint.to_formula(var_string)
    total_formula = And(code_commute_formula, error_formula, nwz_formula)
    total_formula.clausify()
    g = Glucose3()
    for clause in list(total_formula):
        g.add_clause(clause)
    if not g.solve():
        warn("Solve did not succeed.")
    model = g.get_model()
    return var_string.assign_to_stim(model)

if __name__ == "__main__":
    stabilizers = [stim.PauliString("ZZI")]
    code = StabilizerCode.from_stim(stabilizers)
    err = stim.PauliString("IIX")
    new_stabilizer = solve_single_stabilizer(code, err, pad=False)
    print(f"New stabilizer:", new_stabilizer)