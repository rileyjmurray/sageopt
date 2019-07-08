from sageopt import standard_sig_monomials, sig_solrec
from sageopt import conditional_sage_data, sig_constrained_dual, sig_constrained_primal

# Problem data
n = 3
A = standard_sig_monomials(n)
f = 1e4 * sum(A)
main_gts = [
    1e4 + 1e-2 * A[2] / A[0] - 7.0711 / A[0],
    1e4 * (A[2] + A[0]) + 8.54e-3 * (70.7012 * A[2] * (A[0] + A[2])) / A[0]
        - 6.0385e-1 * (A[0] + A[2]) * (1.0 / A[0] + 1.0 / A[1])
]
bounds = [
    1e4 - 1e4 * A[0], 1e4 * A[0] - 1e-4,
    1e4 - 1e4 * A[1], 1e4 * A[1] - 7.0711,
    1e4 - 1e4 * A[2], 1e4 * A[2] - 1e-4,
    A[0] - 69.7107 * A[2], (1e8 * 70.7107 - 1) - A[0] / A[2]
]
gts = main_gts + bounds

# setup SAGE relaxations
X = conditional_sage_data(f, gts, [])
prim = sig_constrained_primal(f, main_gts, [], 0, 1, 0, X)
dual = sig_constrained_dual(f, main_gts, [], 0, 1, 0, X)

# Solve SAGE relaxations, and print the resulting objective values.
prim.solve()
dual.solve()
print('\n')
print(prim.value)
print(dual.value)

# Solution recovery
solns = sig_solrec(dual)
print(f(solns[0]))
