"""Using the eight-qubit system with its two symmetry group generators,
prepare a logical Bell state on the first two logical qubits and measure <Z0Z1>."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cirq
import stim
import stimcirq
import htlogicalgates as htlg
from encoded.utils import get_observables
from encoded.prep_circuit import cb_prep_circuit
from encoded.htlg_interface import stim_pauli_string_to_htlg_str, htlg_circuit_to_stim
from encoded.error_detection import run_with_error_detection
from encoded.noise_models import QubitDependentNoiseGate

stabilizers = [
    stim.PauliString("Z_Z_"),
    stim.PauliString("_Z_Z")
]
logical_ops = get_observables(stabilizers)
logical_zs = [t[1] for t in logical_ops]
logical_xs = [t[0] for t in logical_ops]

with open("four_qubit_circuits.pkl", "rb") as f:
    input_dict= pickle.load(f)

encoding_ckt = input_dict["encoding"]
logical_h = input_dict["logical_h"]
logical_cx = input_dict["logical_cx"]
logical_ckt_stim = logical_h + logical_cx

def noise_circuit(noise_rate: float) -> stim.Circuit:
    stim_ckt = stim.Circuit()
    stim_ckt.append("I", range(4))
    cirq_ckt = stimcirq.stim_circuit_to_cirq_circuit(stim_ckt)
    qs = cirq.LineQubit.range(4)
    noise_map = {q: cirq.DepolarizingChannel(p=noise_rate).on(q) for q in qs}
    noise_gate = QubitDependentNoiseGate(noise_map)
    noisy_cirq_ckt = cirq_ckt.with_noise(noise_gate)
    noisy_stim_ckt = stimcirq.cirq_circuit_to_stim_circuit(noisy_cirq_ckt)
    return noisy_stim_ckt


def stim_bits_to_floats(stim_bits: np.ndarray) -> np.ndarray:
    floats = []
    for bit in stim_bits:
        if bit:
            floats.append(-1.)
        else:
            floats.append(1.)
    return np.array(floats)


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
    bits = run_with_error_detection(total_ckt, stabilizers, logical_xs[0] * logical_xs[1], shots, measure_noise_rate)
    floats = stim_bits_to_floats(bits)
    return np.average(floats)

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
df.to_csv("four_qubit_unmitigated_ckt_level_result.csv")

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
df.to_csv("four_qubit_mitigated_ckt_level_result.csv")

# fig, ax = plt.subplots()
# ax.errorbar(df["noise_rate"], df["mitigated_avg"], yerr=df["mitigated_std"], label="Mitigated")
# ax.errorbar(df["noise_rate"], df["unmitigated_avg"], yerr=df["unmitigated_std"], label="Unmitigated")
# ax.legend()
# plt.show()