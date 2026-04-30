import stim
from encoded.add_stabilizers import build_code_randomly, test_group_membership

def all_single_qubit_errors(n: int):
    all_errors = [stim.PauliString('_' * n)]
    for i in range(n):
        for p in [1, 2, 3]:
            mask = [0] * n
            mask[i] = p
            all_errors.append(stim.PauliString(mask))
    return all_errors

stabilizers = [
    stim.PauliString("XZZXI"),
    stim.PauliString("IXZZX"),
]
errors = all_single_qubit_errors(5)
new_stabilizers = build_code_randomly(stabilizers, errors)
for stab in new_stabilizers:
    print(stab)

for e1 in errors:
    for e2 in errors:
        if e1 != e2:
            e = e1 * e2
            anticommute_tests = [not e.commutes(gen) for gen in new_stabilizers]
            in_group = test_group_membership(e, new_stabilizers)
            assert any(anticommute_tests) or in_group, f"e1 = {e1}, e2 = {e2}"