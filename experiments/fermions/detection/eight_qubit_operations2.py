"""Like the experiment in eight_qubit_operations.py, but with circuit-level
noise and using subspace expansion instead of error detection."""

import pickle
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import stim
import cirq
import stimcirq
import htlogicalgates as htlg
from encoded.utils import get_observables, stim_bits_to_floats
from encoded.prep_circuit import cb_prep_circuit
from encoded.htlg_interface import stim_pauli_string_to_htlg_str, htlg_circuit_to_stim
from encoded.symmetry_expansion import stim_subspace_expansion

stabilizers = [
    stim.PauliString("Z_Z_Z_Z_"),
    stim.PauliString("_Z_Z_Z_Z")
]
logical_ops = get_observables(stabilizers)
logical_zs = [t[1] for t in logical_ops]
logical_xs = [t[0] for t in logical_ops]

with open("eight_qubit_circuits.pkl", "rb") as f:
    input_dict= pickle.load(f)

encoding_ckt = input_dict["encoding"]
logical_h = input_dict["logical_h"]
logical_cx = input_dict["logical_cx"]
logical_ckt_stim = logical_h + logical_cx

def noise_circuit(noise_rate: float) -> stim.Circuit:
    noise_ckt = stim.Circuit()
    noise_ckt.append("Depolarize1", range(8), arg=noise_rate)
    return noise_ckt


def stim_circuit_with_cl_depolarizing_noise(stim_circuit: stim.Circuit, noise_rate: float) -> stim.Circuit:
    # cirq_circuit = stimcirq.stim_circuit_to_cirq_circuit(stim_circuit)
    # noisy_cirq_circuit = cirq_circuit.with_noise(cirq.DepolarizingChannel(p=noise_rate))
    # noisy_stim_circuit = stimcirq.cirq_circuit_to_stim_circuit(noisy_cirq_circuit)
    # return noisy_stim_circuit
    # return stim_circuit + noise_circuit(noise_rate)
    return deepcopy(stim_circuit)


def unmitigated_expectation_value(noise_rate: float, shots: int) -> float:
    noise_ckt = noise_circuit(noise_rate)
    total_ckt = encoding_ckt + noise_ckt + logical_ckt_stim + noise_ckt
    total_ckt.append("MPP", logical_xs[0] * logical_xs[1])
    sampler = total_ckt.compile_sampler()
    bits = sampler.sample(shots)
    floats = stim_bits_to_floats(bits)
    return np.average(floats)


def mitigated_expectation_value(noise_rate: float, measure_noise_rate: float, shots: int) -> float:
    noise_ckt = noise_circuit(noise_rate)
    total_ckt = encoding_ckt + noise_ckt + logical_ckt_stim + noise_ckt
    return stim_subspace_expansion(total_ckt, logical_xs[0] * logical_xs[1], stabilizers, shots)

shots = 10_000
reps = 10
noise_rates = np.linspace(1e-4, 1e-2, num=10)
records = []
for noise_rate in noise_rates:
    all_reps_unmitigated = []
    for _ in range(reps):
        unmitigated_result = unmitigated_expectation_value(noise_rate, shots)
        all_reps_unmitigated.append(unmitigated_result)
    records.append((
        noise_rate,
        np.average(all_reps_unmitigated), np.std(all_reps_unmitigated)
    ))
df = pd.DataFrame.from_records(records, columns=["noise_rate", "avg", "std"])
df.to_csv("eight_qubit_unmitigated_result2.csv")

measure_noise_rates = [0., 0.1]
records = []
for noise_rate in noise_rates:
    for meas_noise_rate in measure_noise_rates:
        all_reps_mitigated = []
        for _ in range(reps):
            mitigated_result = mitigated_expectation_value(noise_rate, meas_noise_rate, shots)
            all_reps_mitigated.append(mitigated_result)
        records.append((
            noise_rate, meas_noise_rate,
            np.average(all_reps_mitigated), np.std(all_reps_mitigated),
        ))
df = pd.DataFrame.from_records(records, columns=["noise_rate", "meas_noise_rate", "avg", "std"])
df.to_csv("eight_qubit_mitigated_result2.csv")

# fig, ax = plt.subplots()
# ax.errorbar(df["noise_rate"], df["mitigated_avg"], yerr=df["mitigated_std"], label="Mitigated")
# ax.errorbar(df["noise_rate"], df["unmitigated_avg"], yerr=df["unmitigated_std"], label="Unmitigated")
# ax.legend()
# plt.show()