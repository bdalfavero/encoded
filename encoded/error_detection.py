from typing import List
from copy import deepcopy
import numpy as np
import stim

def run_with_error_detection(
    circuit: stim.Circuit, stabilizers: List[stim.PauliString], observable: stim.PauliString,
    shots: int
) -> np.ndarray:
    """Measure a the eigenvalues of a Pauli string after executing a circuit and postselecting on
    all stabilizers being measured as zero."""

    total_ckt = deepcopy(circuit)
    for stab in stabilizers:
        total_ckt.append("MPP", stab)
    total_ckt.append("MPP", observable)
    sampler = total_ckt.compile_sampler()
    result = sampler.sample(shots)
    final_measurements = result[:, (result.shape[1] - len(stabilizers) - 1):]
    assert final_measurements.shape[1] == len(stabilizers) + 1

    valid_measurements = [] # Bits from the observable for good shots.
    for row in final_measurements:
        stabilizer_bits = row[:-1]
        observable_bit = row[-1]
        if np.all(np.invert(stabilizer_bits)):
            valid_measurements.append(observable_bit)
    return np.array(valid_measurements)