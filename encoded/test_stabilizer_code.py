import unittest
import numpy as np
import stim
from encoded.stabilizer_code import StabilizerCode

class TestStimConversion(unittest.TestCase):

    def test_repetition_from_stim(self):
        stabilizers = [stim.PauliString("ZZ_"), stim.PauliString("_ZZ")]
        code = StabilizerCode.from_stim(stabilizers)
        target_check_matrix = np.array([
            [False, False, False, True, True, False],
            [False, False, False, False, True, True]
        ])
        check_matrix = code.check_matrix
        self.assertTrue(np.allclose(check_matrix, target_check_matrix))

    def test_repetition_to_stim(self):
        check_matrix = np.array([
            [False, False, False, True, True, False],
            [False, False, False, False, True, True]
        ])
        code = StabilizerCode(check_matrix)
        target_stabilizers = [stim.PauliString("ZZ_"), stim.PauliString("_ZZ")]
        stabilizers = code.to_stim()
        self.assertEqual(target_stabilizers, stabilizers)


class TestPad(unittest.TestCase):

    def test_m0(self):
        stabilizers = [stim.PauliString("ZZ_"), stim.PauliString("_ZZ")]
        code = StabilizerCode.from_stim(stabilizers)
        code.pad(0)
        target_check_matrix = np.array([
            [False, False, False, True, True, False],
            [False, False, False, False, True, True]
        ])
        self.assertTrue(np.allclose(target_check_matrix, code.check_matrix))

    def test_m1(self):
        stabilizers = [stim.PauliString("ZZ_"), stim.PauliString("_ZZ")]
        code = StabilizerCode.from_stim(stabilizers)
        code.pad(1)
        target_check_matrix = np.array([
            [False, False, False, False, True, True, False, False],
            [False, False, False, False, False, True, True, False]
        ])
        self.assertTrue(np.allclose(target_check_matrix, code.check_matrix))

if __name__ == "__main__":
    unittest.main()