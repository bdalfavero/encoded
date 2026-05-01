import numpy as np
import matplotlib.pyplot as plt
import stim
from encoded.symmetry_expansion import stim_subspace_expansion
from encoded.utils import stim_bits_to_floats

stabilizers = [stim.PauliString("ZZ_"), stim.PauliString("_ZZ")]
logical_z = stim.PauliString("Z__")

prep_circuit = stim.Circuit()
prep_circuit.append("CNOT", [0, 1, 0, 2])
logical_x_circuit = stim.Circuit()
logical_x_circuit.append("X", [0, 1, 2])

def noise_circuit(noise_rate: float) -> stim.Circuit:
    noise_ckt = stim.Circuit()
    noise_ckt.append("X_ERROR", [0, 1, 2], arg=noise_rate)
    return noise_ckt

def total_circuit(noise_rate: float) -> stim.Circuit:
    noise_ckt = noise_circuit(noise_rate)
    return prep_circuit + noise_ckt + logical_x_circuit + noise_ckt

def unmitigated_expectation_value(noise_rate: float, shots: int) -> float:
    ckt = total_circuit(noise_rate)
    ckt.append("MPP", logical_z)
    sampler = ckt.compile_sampler()
    result = sampler.sample(shots)
    return np.average(stim_bits_to_floats(result))

def mitigated_expectation_value(noise_rate: float, shots: int) -> float:
    ckt = total_circuit(noise_rate)
    return stim_subspace_expansion(ckt, logical_z, stabilizers, shots)

mitigated_values = []
unmitigated_values = []
shots = 10_000
noise_rates = np.linspace(1e-4, 1e-1, num=10)
for noise_rate in noise_rates:
    mitigated_values.append(mitigated_expectation_value(noise_rate, shots))
    unmitigated_values.append(unmitigated_expectation_value(noise_rate, shots))

fig, ax = plt.subplots()
ax.plot(noise_rates, mitigated_values, label="Mitigated")
ax.plot(noise_rates, unmitigated_values, label="Unmitigated")
ax.legend()
plt.show()