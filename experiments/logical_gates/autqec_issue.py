"""Based on https://github.com/hsayginel/autqec/blob/db614a9bfc91f384d324a647557aa9213c6cdc3b/examples/graph_auts.ipynb"""

from autqec.automorphisms import *
from autqec.graph_auts import *
from autqec.utils.qec import *
from autqec.utils.qiskit import *
from autqec.magma_interface import *
from autqec.ZX_dualities import *
from autqec.ZY_dualities import *

n = 3
k = 1
d = 3
stabilizer_strings = ["ZZI", "IZZ"]
print("Check matrix:")
H_symp = stabs_to_H_symp(stabilizer_strings)
H_3bit = np.hstack([H_symp,(H_symp[:,:n]+H_symp[:,n:])%2])
H_3bit = np.vstack([H_3bit,(H_3bit[0,:]+H_3bit[1,:])%2])
print(H_symp)
print(H_3bit)

auts = valid_clifford_auts(H_3bit, bits_3=True) # computes graph auts that correspond to a physical Clifford operation

circuits = []
symp_mats = []
for num, aut in enumerate(auts):
    phys_act = circ_from_aut(H_symp,aut)        
    phys_circ, _ = phys_act.circ()
    log_act = logical_circ_and_pauli_correct(H_symp,phys_circ)
    circ = log_act.run()
    circuits.append(circ)
    symp_mats.append(log_act.U_logical_act())
print(f"Generated {len(circuits)} circuits.")