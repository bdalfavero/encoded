import unittest
import numpy as np
import cirq
from encoded.expanded_repetition_code import (
    z_bar,
    stabilizer_generators,
    encoding_expanded_repetition
)

class TestExpandedRepetition(unittest.TestCase):

    def test_stabilizers(self):
        prep_ckt, _ = encoding_expanded_repetition('0')
        sim = cirq.Simulator()
        all_expectations = []
        for stabilizer in stabilizer_generators:
            exp_val = sim.simulate_expectation_values(prep_ckt, [stabilizer])[0]
            all_expectations.append(exp_val)
        errs = np.abs(np.array(all_expectations) - 1.0)
        self.assertTrue(np.all(errs <= 1e-4))
    
    def test_logical_z_zero_state(self):
        prep_ckt, _ = encoding_expanded_repetition('0')
        sim = cirq.Simulator()
        exp_val = sim.simulate_expectation_values(prep_ckt, [z_bar])[0]
        err = abs(exp_val - 1.0)
        self.assertTrue(err <= 1e-5)

    def test_logical_z_one_state(self):
        prep_ckt, _ = encoding_expanded_repetition('1')
        sim = cirq.Simulator()
        exp_val = sim.simulate_expectation_values(prep_ckt, [z_bar])[0]
        err = abs(exp_val - -1.0)
        self.assertTrue(err <= 1e-5)

if __name__ == "__main__":
    unittest.main()