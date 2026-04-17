import unittest
import numpy as np
import stim
from encoded.decompose_operators import (
    decompose_operator_to_product,
    decompose_pauli_to_logical_operators
)

class TestDecompose(unittest.TestCase):

    def test_zzi_izz_ziz(self):
        generators = [
            stim.PauliString("ZZI"),
            stim.PauliString("IZZ"),
            stim.PauliString("ZIZ")
        ]
        pstring = stim.PauliString("ZZI")
        pstring_generators = decompose_operator_to_product(pstring, generators)
        target_generators = [stim.PauliString("ZZI")]
        self.assertEqual(target_generators, pstring_generators)
    
    def test_xi_ix(self):
        generators = [
            stim.PauliString("XI"),
            stim.PauliString("IX")
        ]
        pstring = stim.PauliString("XX")
        pstring_generators = decompose_operator_to_product(pstring, generators)
        target_generators = [stim.PauliString("XI"), stim.PauliString("IX")]
        self.assertEqual(target_generators, pstring_generators)

    def test_x_z(self):
        generators = [
            stim.PauliString("X"),
            stim.PauliString("Z")
        ]
        pstring = stim.PauliString("Y")
        pstring_generators = decompose_operator_to_product(pstring, generators)
        target_generators = [stim.PauliString("X"), stim.PauliString("Z")]
        self.assertEqual(target_generators, pstring_generators)

    def test_xi_zi_ix_iz(self):
        generators = [
            stim.PauliString("XI"),
            stim.PauliString("ZI"),
            stim.PauliString("IX"),
            stim.PauliString("IZ")
        ]
        pstring = stim.PauliString("IY")
        pstring_generators = decompose_operator_to_product(pstring, generators)
        target_generators = [stim.PauliString("IX"), stim.PauliString("IZ")]
        self.assertEqual(target_generators, pstring_generators)

    def test_repetition_code_y_bar_ziz(self):
        generators = [
            stim.PauliString("ZZI"),
            stim.PauliString("ZIZ"),
            stim.PauliString("XXX"),
            stim.PauliString("ZII")
        ]
        pstring = stim.PauliString("-XXY")
        target_generators = [
            stim.PauliString("ZIZ"),
            stim.PauliString("XXX"),
            stim.PauliString("ZII")
        ]
        pstring_generators = decompose_operator_to_product(pstring, generators)
        self.assertEqual(target_generators, pstring_generators)


class TestLogicalDecompose(unittest.TestCase):

    def test_repetition_code(self):
        stabilizers = [
            stim.PauliString("ZZI"),
            stim.PauliString("ZIZ")
        ]
        logical_op_dict = [
            (stim.PauliString("X"), stim.PauliString("XXX")),
            (stim.PauliString("Z"), stim.PauliString("ZII"))
        ]
        pstring = stim.PauliString("XYY")
        target_pstring = stim.PauliString("-X")
        actual_pstring = decompose_pauli_to_logical_operators(pstring, logical_op_dict, stabilizers)
        self.assertEqual(target_pstring, actual_pstring)

    def test_repetition_code_y(self):
        stabilizers = [
            stim.PauliString("ZZI"),
            stim.PauliString("ZIZ")
        ]
        logical_op_dict = [
            (stim.PauliString("X"), stim.PauliString("XXX")),
            (stim.PauliString("Z"), stim.PauliString("ZII"))
        ]
        pstring = stim.PauliString("YXX")
        target_pstring = stim.PauliString("Y")
        actual_pstring = decompose_pauli_to_logical_operators(pstring, logical_op_dict, stabilizers)
        self.assertEqual(target_pstring, actual_pstring)

    def test_repetition_code_y_bar_ziz(self):
        stabilizers = [
            stim.PauliString("ZZI"),
            stim.PauliString("ZIZ")
        ]
        logical_op_dict = [
            (stim.PauliString("X"), stim.PauliString("XXX")),
            (stim.PauliString("Z"), stim.PauliString("ZII"))
        ]
        pstring = stim.PauliString("-XXY")
        target_pstring = stim.PauliString("-Y")
        actual_pstring = decompose_pauli_to_logical_operators(pstring, logical_op_dict, stabilizers)
        self.assertEqual(target_pstring, actual_pstring)

if __name__ == "__main__":
    unittest.main()