import unittest
import numpy as np
import stim
from encoded.add_stabilizers import _form_linear_system, add_stabilizer, test_group_membership, knill_laflamme_cost_function

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
            stim.PauliString("ZZ"),
        ]
        errs = [
            stim.PauliString("XX")
        ]
        new_stabilizers = add_stabilizer(stabilizers, errs, verbose=False, extra_support=stim.PauliString("Z"))
        target_stabilizers = [
            stim.PauliString("ZZI"),
            stim.PauliString("ZIZ"),
        ]
        self.assertEqual(new_stabilizers, target_stabilizers)


class TestGroupMembership(unittest.TestCase):

    def test_id(self):
        operator = stim.PauliString("__")
        generators = [
            stim.PauliString("X_"),
            stim.PauliString("_X")
        ]
        self.assertTrue(test_group_membership(operator, generators))

    def test_repetition(self):
        operator = stim.PauliString("Z_Z")
        generators = [
            stim.PauliString("ZZ_"),
            stim.PauliString("_ZZ")
        ]
        self.assertTrue(test_group_membership(operator, generators))


class TestKLCost(unittest.TestCase):

    def test_repetition(self):
        generators = [
            stim.PauliString("ZZ_"),
            stim.PauliString("_ZZ")
        ]
        errors = [
            stim.PauliString("X__"),
            stim.PauliString("_X_"),
            stim.PauliString("__X")
        ]
        weights = [1.] * len(errors)
        loss = knill_laflamme_cost_function(generators, errors, weights)
        self.assertTrue(abs(loss + 3.) <= 1e-12)

    def test_phase_flip(self):
        generators = [
            stim.PauliString("XX_"),
            stim.PauliString("_XX")
        ]
        errors = [
            stim.PauliString("Z__"),
            stim.PauliString("_Z_"),
            stim.PauliString("__Z")
        ]
        weights = [1.] * len(errors)
        loss = knill_laflamme_cost_function(generators, errors, weights)
        self.assertTrue(abs(loss + 3.) <= 1e-12)

    def test_five_qubit(self):
        generators = [
            stim.PauliString("XZZXI"),
            stim.PauliString("IXZZX"),
            stim.PauliString("XIXZZ"),
            stim.PauliString("ZXIXZ"),
        ]
        errors = [
            stim.PauliString("Z____"),
            stim.PauliString("_Z___"),
            stim.PauliString("__Z__"),
            stim.PauliString("___Z_"),
            stim.PauliString("____Z"),
            stim.PauliString("X____"),
            stim.PauliString("_X___"),
            stim.PauliString("__X__"),
            stim.PauliString("___X_"),
            stim.PauliString("____X"),
            stim.PauliString("Y____"),
            stim.PauliString("_Y___"),
            stim.PauliString("__Y__"),
            stim.PauliString("___Y_"),
            stim.PauliString("____Y")
        ]
        weights = [1.] * len(errors)
        loss = knill_laflamme_cost_function(generators, errors, weights)
        self.assertTrue(abs(loss + len(errors)) <= 1e-12)

if __name__ == "__main__":
    unittest.main()