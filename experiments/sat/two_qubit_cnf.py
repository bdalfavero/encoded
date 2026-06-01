from sympy import *
from pysat.formula import Atom, And, XOr

# First index is the string, second is the qubit.
x_11, x_12, z_11, z_12 = symbols("x_11,x_12,z_11,z_12")
x_21, x_22, z_21, z_22 = symbols("x_21,x_22,z_21,z_22")

symplectic_ip = (x_11 & z_21) ^ (x_21 & z_11) ^ (x_12 & z_22) ^ (x_22 & z_12)
print("Original expression:", symplectic_ip)
symplectic_ip_cnf = to_cnf(symplectic_ip)
print("CNF:", symplectic_ip_cnf)