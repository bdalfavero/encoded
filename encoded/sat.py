from typing import List, Tuple
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
            if self._commutes:
                subformulas.append(Neg(XOr(*x_ands, *z_ands)))
            else:
                subformulas.append(XOr(*x_ands, *z_ands))
        return And(*subformulas)

if __name__ == "__main__":
    stabilizers = [stim.PauliString("ZZI"), stim.PauliString("IZZ")]
    code = StabilizerCode.from_stim(stabilizers)
    var_string = VariableString(3, 0)
    comm_constraint = CommutationConstraint(code, True)
    commute_formula = comm_constraint.to_formula(var_string)
    weight_zero_formula = var_string.non_weight_zero_constraint()
    formula = And(weight_zero_formula, commute_formula)
    print(formula)
    formula.clausify()
    print("Atoms:")
    for atom in formula.atoms():
        print(atom.name)
    g = Glucose3()
    print("Clauses:")
    for clause in list(formula):
        print(clause)
        g.add_clause(clause)
    print(f"Solving...")
    print(g.solve())
    model = g.get_model()
    print(model)
    pstring = var_string.assign_to_stim(model)
    print("Found string", pstring)