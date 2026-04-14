import unittest
import numpy as np
from encoded.decompose_operators import _boolean_rref, _boolean_backsub_solve, solve_boolean_system

class TestRREF(unittest.TestCase):

    def test_id(self):
        A = np.eye(3).astype(bool)
        b = np.array([True, False, False])
        A_rref, b_rref = _boolean_rref(A, b)
        self.assertTrue(np.allclose(A_rref, A) and np.allclose(b_rref, b))
    
    def test_one_below_diagonal(self):
        A = np.array([
            [True, False],
            [True, True]
        ])
        b = np.array([True, True])
        A_target = np.eye(2).astype(bool)
        b_target = np.array([True, False])
        A_rref, b_rref = _boolean_rref(A, b)
        self.assertTrue(np.allclose(A_target, A_rref) and np.allclose(b_target, b_rref))
    
    def test_zzi_izz_ziz(self):
        A = np.array([
            [True, False, True],
            [True, True, False],
            [False, True, True]
        ])
        b = np.zeros(3).astype(bool)
        A_rref, b_rref = _boolean_rref(A, b)
        A_target = np.array([
            [True, False, True],
            [False, True, True],
            [False, False, False]
        ])
        self.assertTrue(np.allclose(A_rref, A_target))


class TestBacksub(unittest.TestCase):

    def test_id(self):
        A = np.eye(3).astype(bool)
        b = np.array([True, False, False])
        x = _boolean_backsub_solve(A, b)
        print(x)
        self.assertTrue(np.allclose(x, b))
    
    def test_one_above_diagonal(self):
        A = np.array([
            [True, True],
            [False, True]
        ])
        b = np.array([True, False])
        x = _boolean_backsub_solve(A, b)
        x_target = np.array([True, False])
        self.assertTrue(np.allclose(x, x_target))


class TestLinearSolve(unittest.TestCase):

    def test_id(self):
        A = np.eye(3).astype(bool)
        b = np.array([True, False, False])
        x = solve_boolean_system(A, b)
        self.assertTrue(np.allclose(x, b))

    def test_zzi_izz_ziz(self):
        A = np.array([
            [True, False, True],
            [True, True, False],
            [False, True, True]
        ])
        b = np.array([True, False, True])
        x_target = np.array([True, True, False])
        x = solve_boolean_system(A, b)
        self.assertTrue(np.allclose(x, x_target))

if __name__ == "__main__":
    unittest.main()