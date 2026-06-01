"""Attempt to increase the distance of the [[4,2,2]] code."""

import sys
import stim
import numpy as np
from encoded.stabilizer_code import StabilizerCode
from encoded.sat import increase_distance, enumerate_errors

stabilizers = [stim.PauliString("XXXX"), stim.PauliString("ZZZZ")]
code = StabilizerCode.from_stim(stabilizers)
new_stabilizers = increase_distance(code, 2, 3)
if new_stabilizers is None:
    print("Solve failed.")
    sys.exit(1)

for stabilizer in new_stabilizers:
    print(stabilizer)
all_stabilizers = stabilizers + new_stabilizers

all_errors = enumerate_errors(5, 2)
for err in all_errors:
    comm_tests = [err.commutes(stabilizer) for stabilizer in all_stabilizers]
    assert all(np.invert(comm_tests)), f"Failed for error {err}, tests={comm_tests}"