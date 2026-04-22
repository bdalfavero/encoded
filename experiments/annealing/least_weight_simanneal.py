from random import randint
import numpy as np
from simanneal import Annealer
from encoded.binary_linalg import enumerate_all_solutions

def exact_lowest_weight_solution(A, b):
    all_solutions = enumerate_all_solutions(A, b)
    return min(all_solutions, key=lambda x: sum(x))


class BinarySolveAnnealer(Annealer):

    def move(self):
        # Randomly flip a bit.
        i = randint(0, self.state.size - 1)
        self.state[i] = not self.state[i]

    def energy(self):
        x = self.state
        solution_loss = np.sum(A @ x ^ b) # This should be all false for a correct solution.
        weight_loss = np.sum(x) # Penalize Hamming weight.
        # if solution_loss == 0:
        #     print("With state", x)
        #     print(f"Solution loss is {solution_loss}")
        #     print(f"Weight loss is {weight_loss}")
        return solution_loss + 0.1 * weight_loss


A = np.array([
    [True, True, False],
    [False, False, True]
])
b = np.array([False, True])

x_exact = exact_lowest_weight_solution(A, b)
print("Exact solution:", x_exact)
print("Solution loss:", np.sum(A @ x_exact ^ b))

# Solve with SA.
x0 = np.array([False, False, False])
annealer = BinarySolveAnnealer(x0)
x_sa, loss = annealer.anneal()
print(f"SA found solution", x_sa, f"with loss {loss}.")