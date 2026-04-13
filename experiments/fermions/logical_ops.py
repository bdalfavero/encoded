import stim
from encoded.utils import get_observables

print("Four-qubit code")
generators = [
    stim.PauliString("ZIZII"),
    stim.PauliString("IZIZI"),
    stim.PauliString("IZZIZ")
]
logical_ops = get_observables(generators)
for lop in logical_ops:
    print(lop)
print("Eight-qubit code")
generators = [
    stim.PauliString("ZIZIZIZIII"),
    stim.PauliString("IZIZIZIZII"),
    stim.PauliString("ZZIIZZIIZI"),
    stim.PauliString("ZIZIIZIZIZ")
]
logical_ops = get_observables(generators)
for lop in logical_ops:
    print(lop)