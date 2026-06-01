from typing import List
import numpy as np
import stim
from pysat.formula import Atom, Formula, And, Or, XOr, Neg
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
    formula = comm_constraint.to_formula(var_string)
    print(formula)
    formula.clausify()
    print(list(formula))