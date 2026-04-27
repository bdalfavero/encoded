"""Using the eight-qubit system with its two symmetry group generators,
prepare a logical Bell state on the first two logical qubits and measure <Z0Z1>."""

import numpy as np
import matplotlib.pyplot as plt
import stim
import htlogicalgates as htlg
from encoded.utils import get_observables
from encoded.prep_circuit import cb_prep_circuit
from encoded.htlg_interface import stim_pauli_string_to_htlg_str, htlg_circuit_to_stim
from encoded.error_detection import run_with_error_detection

def stim_pauli_string_to_htlg_str(ps: stim.PauliString) -> str:
    ps_str = ""
    for i, p in enumerate(ps):
        if p == 1:
            ps_str += f"X{i} "
        elif p == 2:
            ps_str += f"Y{i} "
        elif p == 3:
            ps_str += f"Z{i} "
        else:
            continue
    return ps_str[:-1]

stabilizers = [
    stim.PauliString("Z_Z_Z_Z_"),
    stim.PauliString("_Z_Z_Z_Z")
]
logical_ops = get_observables(stabilizers)
logical_zs = [t[1] for t in logical_ops]
logical_xs = [t[0] for t in logical_ops]

stabs_htlg = [stim_pauli_string_to_htlg_str(stab) for stab in stabilizers]
logical_zs_htlg = [stim_pauli_string_to_htlg_str(zbar) for zbar in logical_zs]
logical_xs_htlg = [stim_pauli_string_to_htlg_str(xbar) for xbar in logical_xs]

encoding_ckt = cb_prep_circuit(stabilizers, logical_zs, [False] * 6)
print("Encoding circuit")
print(encoding_ckt)

# Get the physical version of the logical circuit.
logical_gate = htlg.Circuit(6)
logical_gate.h(1)
# logical_gate.cx(0, 1)
stab_code = htlg.StabilizerCode(logical_xs_htlg, logical_zs_htlg, stabs_htlg)
connectivity = htlg.Connectivity("circular", num_qubits=8)
logical_circ, status = htlg.tailor_logical_gate(stab_code, connectivity, logical_gate, num_cz_layers=4)
print(status)
logical_ckt_stim = htlg_circuit_to_stim(logical_circ)
print("Logical circuit")
print(logical_ckt_stim)

def noise_circuit(noise_rate: float) -> stim.Circuit:
    noise_ckt = stim.Circuit()
    noise_ckt.append("Depolarize1", range(8), arg=noise_rate)
    return noise_ckt


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
    total_ckt.append("MPP", logical_zs[0])
    sampler = total_ckt.compile_sampler()
    bits = sampler.sample(shots)
    floats = stim_bits_to_floats(bits)
    return np.average(floats)


def mitigated_expectation_value(noise_rate: float, shots: int) -> float:
    noise_ckt = noise_circuit(noise_rate)
    total_ckt = encoding_ckt + noise_ckt + logical_ckt_stim + noise_ckt
    bits = run_with_error_detection(total_ckt, stabilizers, logical_zs[0], shots)
    floats = stim_bits_to_floats(bits)
    return np.average(floats)

shots = 10_000
noise_rates = np.linspace(1e-4, 1e-2, num=10)
mitigated_results = []
unmitigated_results = []
for noise_rate in noise_rates:
    mitigated_result = mitigated_expectation_value(noise_rate, shots)
    unmitigated_result = unmitigated_expectation_value(noise_rate, shots)
    mitigated_results.append(mitigated_result)
    unmitigated_results.append(unmitigated_result)

fig, ax = plt.subplots()
ax.plot(noise_rates, mitigated_results, label="Mitigated")
ax.plot(noise_rates, unmitigated_results, label="Unmitigated")
ax.legend()
plt.show()