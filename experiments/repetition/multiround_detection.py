import stim
from encoded.error_detection import _multi_round_circuit

stabilizers = [
    stim.PauliString("ZZ_"), stim.PauliString("_ZZ")
]
zbar = stim.PauliString("Z__")
logical_x_ckt = stim.Circuit()
logical_x_ckt.append("X", [0, 1, 2])

ed_ckt = _multi_round_circuit([logical_x_ckt ,logical_x_ckt], stabilizers, zbar, 1e-2)
print(ed_ckt)

sampler = ed_ckt.compile_detector_sampler()
detection_events, observable_flips = sampler.sample(10, separate_observables=True)
print(detection_events)
print(observable_flips)