import unittest
import stim
from encoded.code_extension import encoding_unitary_for_new_stabilizer

class TestEncodingCircuits(unittest.TestCase):

    def test_xiix(self):
        stabilizer = stim.PauliString("XIIX")
        encoding_ckt = encoding_unitary_for_new_stabilizer(stabilizer)
        target_circuit = stim.Circuit()
        target_circuit.append("H", 0)
        target_circuit.append("CNOT", [0, 3])
        target_circuit.append("H", [0, 3])
        self.assertTrue(encoding_ckt == target_circuit)

    def test_iiizz(self):
        stabilizer = stim.PauliString("IIIZZ")
        encoding_ckt = encoding_unitary_for_new_stabilizer(stabilizer)
        target_circuit = stim.Circuit()
        target_circuit.append("CNOT", [3, 4])
        self.assertTrue(encoding_ckt == target_circuit)

if __name__ == "__main__":
    unittest.main()