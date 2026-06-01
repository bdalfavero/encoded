import unittest
import stim
from encoded.stabilizer_code import StabilizerCode
from encoded.sat import solve_single_stabilizer, VariableString, ErrorConstraint

class TestSingleStabilizer(unittest.TestCase):

    def test_zzi_detect_iix(self):
        stabilizers = [stim.PauliString("ZZI")]
        code = StabilizerCode.from_stim(stabilizers)
        err = stim.PauliString("IIX")
        new_stabilizer = solve_single_stabilizer(code, err, pad=False)
        comm_tests = [new_stabilizer.commutes(stabilizer) for stabilizer in stabilizers]
        err_comm_test = not new_stabilizer.commutes(err)
        self.assertTrue(all(comm_tests + [err_comm_test]))
    
    def test_repetition_distance_four(self):
        stabilizers = [stim.PauliString("ZZI"), stim.PauliString("IZZ")]
        code = StabilizerCode.from_stim(stabilizers)
        err = stim.PauliString("XXX")
        new_stabilizer = solve_single_stabilizer(code, err, pad=True)
        mask = list(new_stabilizer)
        # new_stabilizer should have two Z's and two I's.
        num_zs = mask.count(3)
        num_ids = mask.count(0)
        self.assertTrue((num_zs == 2) and (num_ids == 2))


class TestErrorConstraints(unittest.TestCase):

    def test_zzi_xii(self):
        var_string = VariableString(3, 0)
        err = stim.PauliString("XII")
        assignment_string = stim.PauliString("ZZI")
        assignment_ints = var_string.assignment_from_stim(assignment_string)
        formula = ErrorConstraint(err).to_formula([var_string])
        self.assertTrue(formula.satisfied(model=assignment_ints))

    def test_phase_flip_izi(self):
        var_string1 = VariableString(3, 0)
        var_string2 = VariableString(3, 1)
        err = stim.PauliString("IZI")
        assignment_string1 = stim.PauliString("XXI")
        assignment_string2 = stim.PauliString("XIX")
        assignment_ints1 = var_string1.assignment_from_stim(assignment_string1)
        assignment_ints2 = var_string2.assignment_from_stim(assignment_string2)
        assignment_ints = assignment_ints1 + assignment_ints2
        formula = ErrorConstraint(err).to_formula([var_string1, var_string2])
        self.assertTrue(formula.satisfied(model=assignment_ints))

    def test_phase_flip_xii(self):
        var_string1 = VariableString(3, 0)
        var_string2 = VariableString(3, 1)
        err = stim.PauliString("XII")
        assignment_string1 = stim.PauliString("XXI")
        assignment_string2 = stim.PauliString("XIX")
        assignment_ints1 = var_string1.assignment_from_stim(assignment_string1)
        assignment_ints2 = var_string2.assignment_from_stim(assignment_string2)
        assignment_ints = assignment_ints1 + assignment_ints2
        formula = ErrorConstraint(err).to_formula([var_string1, var_string2])
        self.assertFalse(formula.satisfied(model=assignment_ints))

    def test_phase_flip_completion_xii(self):
        var_string1 = VariableString(3, 0)
        err = stim.PauliString("XII")
        assignment_string1 = stim.PauliString("XXI")
        code = StabilizerCode.from_stim([assignment_string1])
        assignment_string2 = stim.PauliString("XIX")
        assignment_ints = var_string1.assignment_from_stim(assignment_string2)
        formula = ErrorConstraint(err, code).to_formula([var_string1])
        self.assertFalse(formula.satisfied(model=assignment_ints))

    def test_bit_flip_completion_xii(self):
        var_string1 = VariableString(3, 0)
        err = stim.PauliString("IXI")
        assignment_string1 = stim.PauliString("ZZI")
        code = StabilizerCode.from_stim([assignment_string1])
        assignment_string2 = stim.PauliString("ZIZ")
        assignment_ints = var_string1.assignment_from_stim(assignment_string2)
        formula = ErrorConstraint(err, code).to_formula([var_string1])
        self.assertTrue(formula.satisfied(model=assignment_ints))

if __name__ == "__main__":
    unittest.main()