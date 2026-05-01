from typing import List
from copy import deepcopy
import numpy as np
import stim

def run_with_error_detection(
    circuit: stim.Circuit, stabilizers: List[stim.PauliString], observable: stim.PauliString,
    shots: int, noise_rate: float = 0.
) -> np.ndarray:
    """Measure a the eigenvalues of a Pauli string after executing a circuit and postselecting on
    all stabilizers being measured as zero."""

    total_ckt = deepcopy(circuit)
    for stab in stabilizers:
        total_ckt.append("MPP", stab, arg=noise_rate)
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


def _multi_round_circuit(
    operations: List[stim.Circuit], stabilizers: List[stim.PauliString], observable: stim.PauliString,
    noise_rate: float=0.
) -> stim.Circuit:
    """Circuit for a multi-round error detection experiment.
    
    Arguments:
    operations - Circuits representing the noisy logical operations.
    stabilizers - The stabilizers of the code.
    observable - A Pauli string to be measured at the end.
    shots - Number of shots to perform.
    noise_rate - The rate at which stabilizer measurements fail."""

    measure_ckt = stim.Circuit()
    for stabilizer in stabilizers:
        measure_ckt.append("MPP", stabilizer, arg=noise_rate)
    
    total_ckt = stim.Circuit()
    total_ckt += measure_ckt
    for j, op in enumerate(operations):
        total_ckt += op
        total_ckt += measure_ckt
        for i in range(len(stabilizers)):
            total_ckt.append_from_stim_program_text(f"DETECTOR rec[{-i - 1}] rec[{-i - 1 - len(stabilizers)}]")
    total_ckt.append("MPP", observable)
    total_ckt.append_from_stim_program_text("OBSERVABLE_INCLUDE(0) rec[-1]")
    return total_ckt


def _filter_results(detection_events: np.ndarray, observable_flips: np.ndarray) -> np.ndarray:
    """Given all of the detection events and the eigenvalues (flips) of a Pauli string,
    keep only the eigenvalues from shots with no detected errors.
    
    e.g. suppose we go the following detectors and eigenvalues:
    [[0, 0], [0, 1], [0, 0]], [0, 1, 0].
    We would return only the eigenvalues [0, 1]."""

    assert observable_flips.size == detection_events.shape[0]
    
    kept_eigenvalues = []
    for i in range(observable_flips.size):
        if np.all(np.invert(detection_events[i, :])):
            kept_eigenvalues.append(observable_flips[i])
    return np.array(kept_eigenvalues)


def multi_round_error_detection(
    operations: List[stim.Circuit], stabilizers: List[stim.PauliString], observable: stim.PauliString,
    shots: int, noise_rate: float=0.
) -> np.ndarray:
    """Perform multiple logical operations, interspersed with rounds of error detection.
    Start by performing transversal initialization with the stabilizers, then
    alternate performing a logical operation and measuring all of the stabilizers.
    
    Arguments:
    operations - Circuits representing the noisy logical operations.
    stabilizers - The stabilizers of the code.
    observable - A Pauli string to be measured at the end.
    shots - Number of shots to perform.
    noise_rate - The rate at which stabilizer measurements fail.

    Returns:
    eigenvalues - The eigenvalues of the observable in binary form."""

    pass