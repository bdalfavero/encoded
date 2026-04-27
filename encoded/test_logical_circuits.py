import unittest
import stim
from encoded.logical_circuits import logical_cnot, logical_hadamard

class TestRepetition(unittest.TestCase):

    def test_repetition(self):
        x_bar1 = stim.PauliString("XXX___")
        z_bar1 = stim.PauliString("Z_____")
        x_bar2 = stim.PauliString("___XXX")
        z_bar2 = stim.PauliString("___Z__")
        cnot_circuit = logical_cnot(z_bar1, x_bar2)
        all_tests = [
            x_bar1.after(cnot_circuit) == x_bar1 * x_bar2,
            x_bar2.after(cnot_circuit) == x_bar2,
            z_bar1.after(cnot_circuit) == z_bar1,
            z_bar2.after(cnot_circuit) == z_bar1 * z_bar2
        ]
        self.assertTrue(all(all_tests))

    # def test_hadamard(self):
    #     x_bar = stim.PauliString("XXX")
    #     z_bar = stim.PauliString("Z__")
    #     h_circuit = logical_hadamard(z_bar, x_bar)
    #     all_tests = [
    #         x_bar.after(h_circuit) == z_bar,
    #         z_bar.after(h_circuit) == x_bar
    #     ]
    #     self.assertTrue(all(all_tests))

class TestSteane(unittest.TestCase):

    def test_steane(self):
        z_bar1 = stim.PauliString("ZZZZZZZ_______")
        z_bar2 = stim.PauliString("_______ZZZZZZZ")
        x_bar1 = stim.PauliString("XXXXXXX_______")
        x_bar2 = stim.PauliString("_______XXXXXXX")
        cnot_circuit = logical_cnot(z_bar1, x_bar2)
        all_tests = [
            x_bar1.after(cnot_circuit) == x_bar1 * x_bar2,
            x_bar2.after(cnot_circuit) == x_bar2,
            z_bar1.after(cnot_circuit) == z_bar1,
            z_bar2.after(cnot_circuit) == z_bar1 * z_bar2
        ]
        self.assertTrue(all(all_tests))

if __name__ == "__main__":
    unittest.main()