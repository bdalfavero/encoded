import unittest
import stim
from encoded.stabilizer_code import StabilizerCode
from encoded.sat import solve_single_stabilizer

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


if __name__ == "__main__":
    unittest.main()