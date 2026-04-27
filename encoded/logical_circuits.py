import stim

def logical_cnot(z_control: stim.PauliString, x_target: stim.PauliString) -> stim.Circuit:
    """Circuit for the logical CNOT."""

    circuit = stim.Circuit()
    for i, pz in enumerate(z_control):
        for j, px in enumerate(x_target):
            if pz == 3 and px == 1:
                circuit.append("CNOT", [i, j])
    return circuit


# def logical_hadamard(z: stim.PauliString, x: stim.PauliString) -> stim.Circuit:
#     """Circuit for the logical Hadamard gate."""

#     circuit = stim.Circuit()
#     for i, pz in enumerate(z):
#         if pz == 3:
#             circuit.append("H", [i])
#     for i, pz in enumerate(z):
#         for j, px in enumerate(x):
#             if pz == 3 and px == 1 and i != j:
#                 circuit.append("CNOT", [i, j])
#     return circuit