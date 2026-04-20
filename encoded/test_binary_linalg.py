import unittest
import numpy as np
import stim
from encoded.binary_linalg import (
    _boolean_rref, _boolean_backsub_solve,
    solve_boolean_system, _pivot_columns
)

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
    
    def test_third_row_linearly_dependent(self):
        A = np.array([
            [True, False, True, False],
            [False, True, True, False],
            [True, False, True, False],
            [False, False, False, True]
        ])
        A_target = ([
            [True, False, True, False],
            [False, True, True, False],
            [False, False, False, True],
            [False, False, False, False]
        ])
        b = np.zeros(4).astype(bool)
        A_rref, _ = _boolean_rref(A, b)
        self.assertTrue(np.allclose(A_rref, A_target))
    
    def test_zzi_izz_e_zii(self):
        A = np.array([
            [True, True, False, False, False, False],
            [False, True, True, False, False, False],
            [True, False, False, False, False, False]
        ])
        b = np.zeros(3).astype(bool)
        A_rref, _ = _boolean_rref(A, b)
        A_target = np.array([
            [True, True, False, False, False, False],
            [False, True, True, False, False, False],
            [False, False, True, False, False, False]
        ])
        self.assertTrue(np.allclose(A_rref, A_target))


class TestBacksub(unittest.TestCase):

    def test_id(self):
        A = np.eye(3).astype(bool)
        b = np.array([True, False, False])
        x = _boolean_backsub_solve(A, b)
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

    def test_zzi_izz_e_zii(self):
        """This tests an under-determined system."""

        A_rref = np.array([
            [True, True, False, False, False, False],
            [False, True, True, False, False, False],
            [False, False, True, False, False, False]
        ])
        b_rref = np.array([False, False, True])
        x_target = np.array([True, True, True, False, False, False])
        x = _boolean_backsub_solve(A_rref, b_rref)
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


class TestPivotColumns(unittest.TestCase):

    def test_id_matrix(self):
        eye = np.eye(4).astype(bool)
        pivot_columns = _pivot_columns(eye)
        target_columns = list(range(4))
        self.assertEqual(set(pivot_columns), set(target_columns))
    
    def test_free_variables(self):
        """This system has one free variable because there is a column at the
        end that is not a pivot."""

        A = np.array([
            [True, False, True, False],
            [False, True, True, True],
            [False, False, True, False]
        ])
        pivot_columns = _pivot_columns(A)
        target_columns = [0, 1, 2]
        self.assertEqual(set(pivot_columns), set(target_columns))
    
    def test_delayed_pivot(self):
        """Sometimes, the pivot columns don't come one after another."""

        A = np.array([
            [True, True, True],
            [False, False, True],
            [False, False, False]
        ])
        pivot_columns = _pivot_columns(A)
        target_columns = [0, 2]
        self.assertEqual(set(pivot_columns), set(target_columns))
    
    def test_tall_matrix(self):
        A = np.zeros((4, 2)).astype(bool)
        A[0, 0] = True
        A[1, 1] = True
        pivot_columns = _pivot_columns(A)
        target_columns = [0, 1]
        self.assertEqual(set(pivot_columns), set(target_columns))

if __name__ == "__main__":
    unittest.main()
