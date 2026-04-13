import stim
from encoded.utils import get_observables

g1 = stim.PauliString("ZZII")
g2 = stim.PauliString("IZZI")
g3 = stim.PauliString("XXXX")
stabilizers = [g1, g2, g3]
logical_ops = get_observables(stabilizers)
for lop in logical_ops:
    print(lop)