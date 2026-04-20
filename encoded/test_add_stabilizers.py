import unittest
import numpy as np
import stim
from encoded.add_stabilizers import _form_linear_system, add_stabilizer

class TestLinearSystem(unittest.TestCase):

    def test_zzi_izz_e_z(self):
        """Test the case where the existing generators are ZZI and IZZ,
        and we want the new stabilizer to anticommte with ZII."""

        stabilizers = [
            stim.PauliString("ZZI"),
            stim.PauliString("IZZ")
        ]
        errs = [
            stim.PauliString("ZII")
        ]
        A, b = _form_linear_system(stabilizers, errs)
        A_target = np.array([
            [True, True, False, False, False, False],
            [False, True, True, False, False, False],
            [True, False, False, False, False, False]
        ])
        b_target = np.array([False, False, True])
        self.assertTrue(np.allclose(A, A_target) and np.allclose(b, b_target))


class TestAddStabilizer(unittest.TestCase):

    def test_zzi_izz_e_z(self):
        """Test the case where the existing generators are ZZI and IZZ,
        and we want the new stabilizer to anticommte with ZII."""

        stabilizers = [
            stim.PauliString("ZZI"),
            stim.PauliString("IZZ")
        ]
        errs = [
            stim.PauliString("ZII")
        ]
        new_stabilizers = add_stabilizer(stabilizers, errs, extra_support=stim.PauliString("X"))
        target_stabilizers = [
            stim.PauliString("ZZII"),
            stim.PauliString("IZZI"),
            stim.PauliString("XXXX")
        ]
        self.assertEqual(new_stabilizers, target_stabilizers)

    def test_zzi_e_iix(self):
        stabilizers = [
            stim.PauliString("ZZI"),
        ]
        errs = [
            stim.PauliString("XXI"),
            stim.PauliString("IIX")
        ]
        new_stabilizers = add_stabilizer(stabilizers, errs, verbose=True)
        target_stabilizers = [
            stim.PauliString("ZZI"),
            stim.PauliString("IZZ"),
        ]
        self.assertEqual(new_stabilizers, target_stabilizers)

if __name__ == "__main__":
    unittest.main()