"""Using the ten-qubit system with its two symmetry group generators and extra stabilizers,
prepare a logical Bell state on the first two logical qubits and measure <Z0Z1>."""

from typing import List, Dict
import pickle
import numpy as np
import matplotlib.pyplot as plt
import stim
from encoded.utils import get_observables
from encoded.error_detection import run_with_error_detection

stabilizers = [
    stim.PauliString("ZIZIZIZIII"),
    stim.PauliString("IZIZIZIZII"),
    stim.PauliString("ZZIIZZIIZI"),
    stim.PauliString("ZIZIIZIZIZ")
]
logical_ops = get_observables(stabilizers)
logical_zs = [t[1] for t in logical_ops]
logical_xs = [t[0] for t in logical_ops]


def noise_circuit(noise_rate_dict: Dict[int, float]) -> stim.Circuit:
    noise_ckt = stim.Circuit()
    for i, noise_rate in noise_rate_dict.items():
        noise_ckt.append("X_ERROR", i, arg=noise_rate)
    return noise_ckt


def stim_bits_to_floats(stim_bits: np.ndarray) -> np.ndarray:
    floats = []
    for bit in stim_bits:
        if bit:
            floats.append(-1.)
        else:
            floats.append(1.)
    return np.array(floats)


def unmitigated_expectation_value(noise_rate_dict: Dict[int, float], shots: int) -> float:
    noise_ckt = noise_circuit(noise_rate_dict)
    total_ckt = encoding_ckt + noise_ckt + logical_ckt_stim + noise_ckt
    total_ckt.append("MPP", logical_zs[0])
    sampler = total_ckt.compile_sampler()
    bits = sampler.sample(shots)
    floats = stim_bits_to_floats(bits)
    return np.average(floats)


def mitigated_expectation_value(noise_rate_dict: Dict[int, float], shots: int) -> float:
    noise_ckt = noise_circuit(noise_rate_dict)
    total_ckt = encoding_ckt + noise_ckt + logical_ckt_stim + noise_ckt
    bits = run_with_error_detection(total_ckt, stabilizers, logical_zs[0], shots)
    floats = stim_bits_to_floats(bits)
    return np.average(floats)

shots = 10_000
noise_rates = np.linspace(1e-4, 1e-2, num=10)
mitigated_results = []
unmitigated_results = []
for noise_rate in noise_rates:
    noise_rate_dict = {}
    for i in range(10):
        if i <= 7:
            noise_rate_dict[i] = noise_rate
        else:
            noise_rate_dict[i] = 0.

    mitigated_result = mitigated_expectation_value(noise_rate_dict, shots)
    unmitigated_result = unmitigated_expectation_value(noise_rate_dict, shots)
    mitigated_results.append(mitigated_result)
    unmitigated_results.append(unmitigated_result)

fig, ax = plt.subplots()
ax.plot(noise_rates, mitigated_results, label="Mitigated")
ax.plot(noise_rates, unmitigated_results, label="Unmitigated")
ax.legend()
plt.show()