import unittest
import stim
import htlogicalgates as htlg
from encoded.htlg_interface import htlg_circuit_to_stim

class TestConversion(unittest.TestCase):

    def test_hadamard(self):
        htlg_circuit = htlg.Circuit(1)
        htlg_circuit.h(0)
        stim_target = stim.Circuit()
        stim_target.append("H", [0])
        stim_actual = htlg_circuit_to_stim(htlg_circuit)
        self.assertEqual(stim_actual, stim_target)

    def test_sxdg(self):
        htlg_circuit = htlg.Circuit(1)
        htlg_circuit.sxdg(0)
        stim_target = stim.Circuit()
        stim_target.append("SQRT_X_DAG", [0])
        stim_actual = htlg_circuit_to_stim(htlg_circuit)
        self.assertEqual(stim_actual, stim_target)

    def test_c_xyz(self):
        htlg_circuit = htlg.Circuit(2)
        htlg_circuit.c_xyz(1)
        stim_target = stim.Circuit()
        stim_target.append("S_DAG", [1])
        stim_target.append("H", [1])
        stim_actual = htlg_circuit_to_stim(htlg_circuit)
        self.assertEqual(stim_actual, stim_target)

    def test_xyz(self):
        htlg_circuit = htlg.Circuit(1)
        htlg_circuit.x(0)
        htlg_circuit.y(0)
        htlg_circuit.z(0)
        stim_target = stim.Circuit()
        stim_target.append("X", [0])
        stim_target.append("Y", [0])
        stim_target.append("Z", [0])
        stim_actual = htlg_circuit_to_stim(htlg_circuit)
        self.assertEqual(stim_actual, stim_target)

    def test_cnot(self):
        htlg_circuit = htlg.Circuit(2)
        htlg_circuit.cx(0, 1)
        stim_target = stim.Circuit()
        stim_target.append("CNOT", [0, 1])
        stim_actual = htlg_circuit_to_stim(htlg_circuit)
        self.assertEqual(stim_actual, stim_target)

    def test_cphase(self):
        htlg_circuit = htlg.Circuit(2)
        htlg_circuit.cz(1, 0)
        stim_target = stim.Circuit()
        stim_target.append("CZ", [1, 0])
        stim_actual = htlg_circuit_to_stim(htlg_circuit)
        self.assertEqual(stim_actual, stim_target)

    def test_cphase(self):
        htlg_circuit = htlg.Circuit(2)
        htlg_circuit.cz(1, 0)
        stim_target = stim.Circuit()
        stim_target.append("CZ", [1, 0])
        stim_actual = htlg_circuit_to_stim(htlg_circuit)
        self.assertEqual(stim_actual, stim_target)

if __name__ == "__main__":
    unittest.main()