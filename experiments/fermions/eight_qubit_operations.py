import stim
from encoded.utils import get_observables
from encoded.prep_circuit import cb_prep_circuit
from encoded.utils import stim_pauli_string_to_circuit
from encoded.logical_circuits import logical_cnot

stabilizers = [
    stim.PauliString("Z_Z_Z_Z_"),
    stim.PauliString("_Z_Z_Z_Z")
]
logical_ops = get_observables(stabilizers)
# for x, z in logical_ops:
#     print("x=", x)
#     print("z=", z)
logical_zs = [t[1] for t in logical_ops]
logicals_xs = [t[0] for t in logical_ops]
logical_x_circuits = [stim_pauli_string_to_circuit(xbar) for xbar in logicals_xs]

total_ckt = stim.Circuit()
zero_prep_ckt = cb_prep_circuit(stabilizers, logical_zs, [False] * len(logical_zs))
total_ckt += zero_prep_ckt
for i in range(len(logicals_xs)):
    if i % 2 == 0:
        total_ckt += logical_x_circuits[i]