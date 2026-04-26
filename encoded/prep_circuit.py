from typing import List
import stim

def cb_prep_circuit(
    generators: List[stim.PauliString], logical_zs: List[stim.PauliString], cb_bools: List[bool]
) -> stim.Circuit:
    """Circuit to prepare a logical computational basis state."""

    assert len(logical_zs) == len(cb_bools)

    signed_zs = []
    for cb_bool, logical_z in zip(cb_bools, logical_zs):
        if cb_bool:
            signed_zs.append(-1 * logical_z)
        else:
            signed_zs.append(logical_z)
    tableau = stim.Tableau.from_stabilizers(generators + signed_zs)
    return tableau.to_circuit()