import unittest
import numpy as np
import stim
from encoded.binary_linalg import (
    _boolean_rref, _boolean_backsub_solve,
    solve_boolean_system, _pivot_columns,
    _pivot_locations, _single_row_backsub,
    _enumerate_bitstrings,
    solve_with_known_values, enumerate_all_solutions,
    system_has_solutions
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
    
    def test_zzi_add_stabilizer(self):
        A = np.array([
            [True, True, False, False, False, False],
            [False, False, False, True, True, False],
            [False, False, False, False, False, True]
        ])
        b = np.array([False, True, True])
        A_rref_target = np.array([
            [True, True, False, False, False, False],
            [False, False, False, True, True, False],
            [False, False, False, False, False, True]
        ])
        b_rref_target = np.array([False, True, True])
        A_rref, b_rref = _boolean_rref(A, b)
        self.assertTrue(np.allclose(A_rref_target, A_rref) and np.allclose(b_rref_target, b_rref))


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


class TestPivots(unittest.TestCase):

    def test_id_matrix(self):
        eye = np.eye(4).astype(bool)
        pivots = _pivot_locations(eye)
        target_pivots = [(i, i) for i in range(4)]
        self.assertEqual(set(pivots), set(target_pivots))
    
    def test_free_variables(self):
        """This system has one free variable because there is a column at the
        end that is not a pivot."""

        A = np.array([
            [True, False, True, False],
            [False, True, True, True],
            [False, False, True, False]
        ])
        pivots = _pivot_locations(A)
        target_pivots = [(0, 0), (1, 1), (2, 2)]
        self.assertEqual(set(pivots), set(target_pivots))
    
    def test_delayed_pivot(self):
        """Sometimes, the pivot columns don't come one after another."""

        A = np.array([
            [True, True, True],
            [False, False, True],
            [False, False, False]
        ])
        pivots = _pivot_locations(A)
        target_pivots = [(0, 0), (1, 2)]
        self.assertEqual(set(pivots), set(target_pivots))
    
    def test_tall_matrix(self):
        A = np.zeros((4, 2)).astype(bool)
        A[0, 0] = True
        A[1, 1] = True
        pivots = _pivot_locations(A)
        target_pivots = [(0, 0), (1, 1)]
        self.assertEqual(set(pivots), set(target_pivots))


class TestSingleRow(unittest.TestCase):

    def test_no_known_rhs_false(self):
        i = 3
        row = np.zeros(5).astype(bool)
        row[i] = True
        known = {}
        rhs = False
        x_i = _single_row_backsub(row, i, known, rhs)
        self.assertTrue(not x_i)

    def test_no_known_rhs_true(self):
        i = 3
        row = np.zeros(5).astype(bool)
        row[i] = True
        known = {}
        rhs = True
        x_i = _single_row_backsub(row, i, known, rhs)
        self.assertTrue(x_i)
    
    def test_one_known(self):
        row = np.zeros(5).astype(bool)
        i = 1
        row[i] = True
        known = {3: True}
        for k in known.keys():
            row[k] = True
        rhs = False
        x_i = _single_row_backsub(row, i, known, rhs)
        self.assertTrue(x_i)

    def test_two_known(self):
        row = np.zeros(5).astype(bool)
        i = 0
        row[i] = True
        known = {3: True, 4: True}
        for k in known.keys():
            row[k] = True
        rhs = False
        x_i = _single_row_backsub(row, i, known, rhs)
        self.assertTrue(not x_i)

    def test_two_known_plus_extra(self):
        """In this case we have a known value that has False in this row."""

        row = np.zeros(5).astype(bool)
        i = 0
        row[i] = True
        known = {1: False, 3: True, 4: False}
        for k in known.keys():
            if k != 1:
                row[k] = True
        rhs = False
        x_i = _single_row_backsub(row, i, known, rhs)
        self.assertTrue(x_i)


class SolveKnown(unittest.TestCase):

    def test_eye_one_unknown(self):
        A = np.eye(5).astype(bool)
        b = np.array([True, False, True, False, True])
        known = {
            1: False,
            2: True,
            3: False,
            4: True
        }
        x = solve_with_known_values(A, b, known)
        self.assertTrue(np.allclose(x, b))

    def test_eye_two_unknowns(self):
        A = np.eye(5).astype(bool)
        b = np.array([True, False, True, False, True])
        known = {
            1: False,
            2: True,
            3: False
        }
        x = solve_with_known_values(A, b, known)
        self.assertTrue(np.allclose(x, b))

    def test_last_variable_free(self):
        A = np.array([
            [True, False, True],
            [False, True, True],
            [False, False, False]
        ])
        b = np.array([True, True, True])
        known = {
            2: True
        }
        x = solve_with_known_values(A, b, known)
        x_target = np.array([False, False, True])
        self.assertTrue(np.allclose(x, x_target))

    def test_middle_column_free(self):
        A = np.array([
            [True, True, False],
            [False, False, True],
        ])
        b = np.array([True, False])
        known = {
            1: False
        }
        x = solve_with_known_values(A, b, known)
        x_target = np.array([True, False, False])
        self.assertTrue(np.allclose(x, x_target))

    def test_tall_system(self):
        A = np.zeros((5, 2)).astype(bool)
        A[0, 0] = True
        A[0, 1] = True
        A[1, 1] = True
        b = np.array([True, False, True, False, False])
        known = {
            2: True, 3: False, 4: False
        }
        x = solve_with_known_values(A, b, known)
        x_target = np.array([True, False])
        self.assertTrue(np.allclose(x, x_target))

class TestAllSolutions(unittest.TestCase):

    def test_eye(self):
        A = np.eye(3)
        b = np.array([True, True, False])
        solutions = enumerate_all_solutions(A, b)
        self.assertTrue(len(solutions) == 1 and np.allclose(b, solutions[0]))

    def test_middle_column_free(self):
        A = np.array([
            [True, True, False],
            [False, False, True],
        ])
        b = np.array([True, False])
        target_solutions = [
            np.array([True, False, False]),
            np.array([False, True, False]),
        ]
        solutions = enumerate_all_solutions(A, b)
        all_tests = []
        for s1, s2 in zip(target_solutions, solutions):
            all_tests.append(np.allclose(s1, s2))
        self.assertTrue(all(all_tests))

    def test_two_free_variables(self):
        A = np.array([
            [True, False, False, True],
            [False, True, True, False],
            [False, False, False, False]
        ])
        b = np.array([True, False, False])
        target_solutions = [
            np.array([True, False, False, False]),
            np.array([False, False, False, True]),
            np.array([True, True, True, False]),
            np.array([False, True, True, True])
        ]
        solutions = enumerate_all_solutions(A, b)
        all_tests = []
        for s1, s2 in zip(target_solutions, solutions):
            all_tests.append(np.allclose(s1, s2))
        self.assertTrue(all(all_tests))


class TestEnumerateBitstrings(unittest.TestCase):

    def test_one_bit(self):
        bitstrings = _enumerate_bitstrings(1)
        targets = [np.array([False]), np.array(True)]
        self.assertEqual(targets, bitstrings)

    def test_two_bits(self):
        bitstrings = _enumerate_bitstrings(2)
        targets = [
            np.array([False, False]),
            np.array([False, True]),
            np.array([True, False]),
            np.array([True, True])
        ]
        all_equal = []
        for b1, b2 in zip(bitstrings, targets):
            all_equal.append(np.allclose(b1, b2))
        self.assertTrue(all(all_equal))


class TestHasSolutions(unittest.TestCase):

    def test_id(self):
        A = np.eye(4).astype(bool)
        b = np.array([True, False, True, True])
        A_rref, b_rref = _boolean_rref(A, b)
        has_solns = system_has_solutions(A_rref, b_rref)
        self.assertTrue(has_solns)

    def test_upper_triangular(self):
        A = np.array([
            [True, True],
            [False, True]
        ])
        b = np.array([True, False])
        A_rref, b_rref = _boolean_rref(A, b)
        has_solns = system_has_solutions(A_rref, b_rref)
        self.assertTrue(has_solns)
    
    def test_bad_b(self):
        A = np.array([
            [True, True],
            [False, False]
        ])
        b = np.array([True, True])
        A_rref, b_rref = _boolean_rref(A, b)
        has_solns = system_has_solutions(A_rref, b_rref)
        self.assertFalse(has_solns)

if __name__ == "__main__":
    unittest.main()
