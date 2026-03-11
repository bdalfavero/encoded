import unittest
import cirq
import stim
from encoded.utils import stim_pauli_string_to_cirq, cirq_pauli_string_to_stim

class TestStimPauliToCirq(unittest.TestCase):

    def test_xxx(self):
        stim_pauli = stim.PauliString("XXX")
        qs = cirq.LineQubit.range(3)
        cirq_pauli_target = cirq.X.on(qs[0]) * cirq.X.on(qs[1]) * cirq.X.on(qs[2])
        cirq_pauli_generated = stim_pauli_string_to_cirq(stim_pauli)
        self.assertTrue(cirq_pauli_generated == cirq_pauli_target)

    def test_minus_zizx(self):
        stim_pauli = stim.PauliString("-ZIZX")
        qs = cirq.LineQubit.range(4)
        cirq_pauli_target = -1 * cirq.Z.on(qs[0])  * cirq.Z.on(qs[2]) * cirq.X.on(qs[3])
        cirq_pauli_generated = stim_pauli_string_to_cirq(stim_pauli)
        self.assertTrue(cirq_pauli_generated == cirq_pauli_target)

    def test_i_iiiyz(self):
        stim_pauli = stim.PauliString("iIIIYZ")
        qs = cirq.LineQubit.range(5)
        cirq_pauli_target = 1j * cirq.Y.on(qs[3])  * cirq.Z.on(qs[4])
        cirq_pauli_generated = stim_pauli_string_to_cirq(stim_pauli)
        self.assertTrue(cirq_pauli_generated == cirq_pauli_target)


class TestCirqPauliToStim(unittest.TestCase):

    def test_xxx(self):
        qs = cirq.LineQubit.range(3)
        cirq_pauli = cirq.X.on(qs[0]) * cirq.X.on(qs[1]) * cirq.X.on(qs[2])
        stim_pauli_target = stim.PauliString("XXX")
        stim_pauli_generated = cirq_pauli_string_to_stim(cirq_pauli)
        self.assertTrue(stim_pauli_generated == stim_pauli_target)

    def test_minus_zizx(self):
        qs = cirq.LineQubit.range(4)
        cirq_pauli = -1 * cirq.Z.on(qs[0])  * cirq.Z.on(qs[2]) * cirq.X.on(qs[3])
        stim_pauli_target = stim.PauliString("-ZIZX")
        stim_pauli_generated = cirq_pauli_string_to_stim(cirq_pauli, qs=qs)
        self.assertTrue(stim_pauli_generated == stim_pauli_target)

    def test_i_iiiyz(self):
        qs = cirq.LineQubit.range(5)
        cirq_pauli = 1j * cirq.Y.on(qs[3])  * cirq.Z.on(qs[4])
        stim_pauli_target = stim.PauliString("iIIIYZ")
        stim_pauli_generated = cirq_pauli_string_to_stim(cirq_pauli, qs=qs)
        self.assertTrue(stim_pauli_generated == stim_pauli_target)

    def test_minus_i_xyxz(self):
        qs = cirq.LineQubit.range(4)
        cirq_pauli = -1j * cirq.X.on(qs[0]) * cirq.Y.on(qs[1]) * cirq.X.on(qs[2]) * cirq.Z.on(qs[3])
        stim_pauli_target = stim.PauliString("-iXYXZ")
        stim_pauli_generated = cirq_pauli_string_to_stim(cirq_pauli, qs=qs)
        self.assertTrue(stim_pauli_generated == stim_pauli_target)

if __name__ == "__main__":
    unittest.main()