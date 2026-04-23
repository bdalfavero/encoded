from random import randint
import stim
import numpy as np
from simanneal import Annealer
from encoded.add_stabilizers import knill_laflamme_cost_function

stabilizers = [
    stim.PauliString("ZZI")
]
errors = [
    stim.PauliString("X__"),
    stim.PauliString("_X_"),
    stim.PauliString("__X")
]
weights = [1.] * len(errors)

class CodeCompletionAnnealer(Annealer):

    def move(self):
        # Randomly flip a bit.
        i = randint(0, self.state.size - 1)
        self.state[i] = not self.state[i]
    
    def energy(self):
        nq = len(self.state) // 2
        xs = self.state[:nq]
        zs = self.state[nq:]
        new_stabilizer = stim.PauliString.from_numpy(xs=xs, zs=zs)
        qec_loss = knill_laflamme_cost_function(stabilizers + [new_stabilizer], errors, weights)
        commutation_tests = [new_stabilizer.commutes(stab) for stab in stabilizers]
        commutation_loss = -1. * np.sum(commutation_tests)
        return qec_loss + commutation_loss

x0 = np.zeros(6).astype(bool)
annealer = CodeCompletionAnnealer(x0)
x_sa, loss = annealer.anneal()
print(f"SA found solution", x_sa, f"with loss {loss}.")