from typing import List, Optional
import numpy as np
from cirq import Simulator, Circuit, PauliSum
import cirq

def symmetry_expansion(
    ckt: Circuit, observable: PauliSum, group_ops: List[PauliSum], sim=Simulator(),
    shots: Optional[int]=None, return_terms=False
) -> float:
    """Do symmetry expansion by summing over group elements."""

    # sim = Simulator()
    expectations = [] # Expectation of OG for G in group.
    denominators = [] # Expectation of G for G in group.
    for g in group_ops:
        if shots is None:
            expectation = sim.simulate_expectation_values(ckt, [observable * g])[0]
            denominator = sim.simulate_expectation_values(ckt, [g])[0]
        else:
            expectation_result = sim.sample_expectation_values(ckt, [observable * g], num_samples=shots)
            expectation = expectation_result[0][0]
            denominator_result = sim.sample_expectation_values(ckt, [g], num_samples=shots)
            denominator = denominator_result[0][0]
        expectations.append(expectation)
        denominators.append(denominator)
    if return_terms:
        return sum(expectations).real / sum(denominators).real, expectations, denominators
    else:
        return sum(expectations).real / sum(denominators).real

if __name__ == "__main__":
    qs = cirq.LineQubit.range(2)
    ckt = Circuit()
    ckt.append(cirq.H(qs[0]))
    ckt.append(cirq.CNOT(qs[0], qs[1]))
    g1 = cirq.X.on(qs[0]) * cirq.X.on(qs[1])
    g2 = cirq.Z.on(qs[0]) * cirq.Z.on(qs[1])
    group_ops = [g1, g2]
    observable = g1 * g2

    sim = cirq.Simulator()
    original_exp_val = sim.simulate_expectation_values(ckt, [observable])[0]
    expansion_exp_val = symmetry_expansion(ckt, observable, group_ops)
    assert abs(original_exp_val - expansion_exp_val) <= 1e-6