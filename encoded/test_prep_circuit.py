import unittest
import numpy as np
import stim
from encoded.prep_circuit import cb_prep_circuit

class TestRepetition(unittest.TestCase):

    def test_zero_bar(self):
        logical_zs = [stim.PauliString("Z__")]
        stabilizers = [
            stim.PauliString("ZZ_"),
            stim.PauliString("_ZZ")
        ]
        prep_ckt = cb_prep_circuit(stabilizers, logical_zs, [False])
        prep_ckt.append("M", [0, 1, 2])
        sampler = prep_ckt.compile_sampler()
        result = sampler.sample(shots=1)
        self.assertTrue(np.all(np.invert(result)))

    def test_one_bar(self):
        logical_zs = [stim.PauliString("Z__")]
        stabilizers = [
            stim.PauliString("ZZ_"),
            stim.PauliString("_ZZ")
        ]
        prep_ckt = cb_prep_circuit(stabilizers, logical_zs, [True])
        prep_ckt.append("M", [0, 1, 2])
        sampler = prep_ckt.compile_sampler()
        result = sampler.sample(shots=1)
        self.assertTrue(np.all(result))

    def test_stabilizers(self):
        logical_zs = [stim.PauliString("Z__")]
        stabilizers = [
            stim.PauliString("ZZ_"),
            stim.PauliString("_ZZ")
        ]
        prep_ckt = cb_prep_circuit(stabilizers, logical_zs, [True])
        prep_ckt.append_from_stim_program_text("MPP Z0*Z1")
        prep_ckt.append_from_stim_program_text("MPP Z1*Z2")
        sampler = prep_ckt.compile_sampler()
        result = sampler.sample(shots=1)
        self.assertTrue(np.all(np.invert(result)))

if __name__ == "__main__":
    unittest.main()