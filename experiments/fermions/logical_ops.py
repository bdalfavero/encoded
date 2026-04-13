import stim
from encoded.utils import get_observables

generators = [
    stim.PauliString("ZIZIZIZIII"),
    stim.PauliString("IZIZIZIZII"),
    stim.PauliString("ZZIIZZIIZI"),
    stim.PauliString("ZIZIIZIZIZ")
]
logical_ops = get_observables(generators)
for lop in logical_ops:
    print(lop)