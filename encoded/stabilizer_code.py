from typing import List
import stim
import numpy as np

class StabilizerCode:

    def __init__(self, check_matrix: np.ndarray):
        assert check_matrix.shape[1] % 2 == 0
        self._check_matrix = check_matrix
    
    @property
    def check_matrix(self):
        return self._check_matrix
    
    @check_matrix.setter
    def check_matrix(self, new_check_matrix):
        self._check_matrix = new_check_matrix
    
    @property
    def n(self):
        return self.check_matrix.shape[1] // 2

    def from_stim(stabilizers: List[stim.PauliString]):
        binary_vecs = []
        for stabilizer in stabilizers:
            xs, zs = stabilizer.to_numpy()
            binary_vecs.append(np.hstack((xs, zs)))
        check_matrix = np.vstack(tuple(binary_vecs))
        return StabilizerCode(check_matrix)
    
    def to_stim(self) -> List[stim.PauliString]:
        stabilizers = []
        for row in self.check_matrix:
            xs, zs = row[:self.n], row[self.n:]
            stabilizers.append(stim.PauliString.from_numpy(xs=xs, zs=zs))
        return stabilizers
    
    def pad(self, m: int):
        """Pad the Pauli strings with m identity operators at the end, e.g.
        <ZZI, IZZ> -> <ZZIII, IZZII> for m = 2."""

        xs = self.check_matrix[:, :self.n]
        zs = self.check_matrix[:, self.n:]
        if m == 0:
            new_xs = xs
            new_zs = zs
        elif m > 0:
            zeros_pad = np.zeros((self.check_matrix.shape[0], m))
            new_xs = np.hstack((xs, zeros_pad))
            new_zs = np.hstack((zs, zeros_pad))
        else:
            raise ValueError(f"m > 0 is required, but m = {m}")
        self.check_matrix = np.hstack((new_xs, new_zs))