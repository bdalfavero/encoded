"""Use sympy to derive the conjunctive normal form of the symplectic inner product of two
boolean vectors. For two vectors [x_1 z_1] and [x_2 z_2], their symplectic inner product is
x_1 z_1 ^ x_2 z_2, where ^ denotes exclusive OR."""

from sympy import *

x1, z1, x2, z2 = symbols("x1,z1,x2,z2")
symplectic_ip = (x1 & z1) ^ (x2 & z2)
print("Original expression:", symplectic_ip)
symplectic_ip_cnf = to_cnf(symplectic_ip)
print("CNF:", symplectic_ip_cnf)