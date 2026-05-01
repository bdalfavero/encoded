from typing import List, Optional
import numpy as np
from cirq import Simulator, Circuit, PauliSum
import cirq
import stim
from encoded.add_stabilizers import generate_stabilizer_elements
from encoded.utils import stim_bits_to_floats

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


def stim_subspace_expansion(
    circuit: stim.Circuit, observable: stim.PauliString, generators: List[stim.PauliString], shots: int
) -> float:
    """Get the expectation value of an observable using symmetry expansion and a stim circuit."""

    def _exp_val(op: stim.PauliString) -> float:
        if op == stim.PauliString("_" * len(op)):
            # Stim can't measure the identity with MPP, apparently.
            return 1.
        else:
            total_ckt = stim.Circuit()
            total_ckt += circuit
            total_ckt.append("MPP", op)
            sampler = total_ckt.compile_sampler()
            results = sampler.sample(shots)
            result_floats = stim_bits_to_floats(results)
            return np.average(result_floats)
    
    group_elements = generate_stabilizer_elements(generators)
    numerators = []
    denominators = []
    for elem in group_elements:
        numerator = _exp_val(observable * elem)
        denominator = _exp_val(elem)
        numerators.append(numerator)
        denominators.append(denominator)
    # print(f"numerator = {np.sum(numerators)}")
    # print(f"denominator = {np.sum(denominators)}")
    return np.sum(numerators) / np.sum(denominators)

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