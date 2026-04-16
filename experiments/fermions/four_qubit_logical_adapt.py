import pickle
import numpy as np
from scipy.sparse import csc_matrix
import openfermion as of
from quimb.tensor.tensor_1d import MatrixProductState
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.pools import FullPauliPool
from adaptvqe.matrix_tools import ket_to_vector
from adaptvqe.tensor_helpers import computational_basis_mps

class QubOpHamiltonian:

    def __init__(self, operator, ref_det, ref_state):
        self.description = "QubOp Hamiltonian"
        self.operator = operator
        self.n = of.utils.count_qubits(operator)
        assert len(ref_det) == self.n
        self.ref_det = ref_det
        self.ref_state = csc_matrix(ref_state).transpose()
        self.tn_ref_state = computational_basis_mps(self.ref_det)
        self._ground_energy = None
        self._ground_state = None
    
    @property
    def ground_state(self):
        if self._ground_state is None:
            ground_energy, ground_state = of.get_ground_state(
                of.get_sparse_operator(self.operator)
            )
            self._ground_state = ground_state
            self._ground_energy = ground_energy

        return self._ground_state

    @property
    def ground_energy(self):
        """
        Returns the exact ground energy of the Hamiltonian.
        """

        if self._ground_energy is None:
            ground_energy, ground_state = of.get_ground_state(
                of.get_sparse_operator(self.operator)
            )
            self._ground_state = ground_state
            self._ground_energy = ground_energy

        return self._ground_energy
                
with open("four_qubit_logical_hamiltonian.pkl", "rb") as f:
    logical_ham = pickle.load(f)

ref_det = [0, 0]
ref_state = ket_to_vector(ref_det)

h = QubOpHamiltonian(logical_ham, ref_det, ref_state)
exact_energy = h.ground_energy
ground_state = h.ground_state
print("Exact energy", exact_energy)
print(ground_state)
print(h.ref_state)

pool = FullPauliPool(n=h.n)
my_adapt = LinAlgAdapt(
    pool=pool,
    custom_hamiltonian=h,
    verbose=False,
    threshold=10**-5,
    max_adapt_iter=5,
    max_opt_iter=10000,
    sel_criterion="gradient",
    recycle_hessian=False,
    rand_degenerate=True
)
my_adapt.run()
data = my_adapt.data