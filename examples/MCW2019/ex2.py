from sageopt import standard_sig_monomials, sig_solrec
from sageopt import infer_domain, sig_constrained_relaxation

# Problem data
n = 3
A = standard_sig_monomials(n)
f = 1e4 * sum(A)
main_gts = [
    1e4 + 1e-2 * A[2] / A[0] - 7.0711 / A[0],
    1e4 * (A[2] + A[0]) + 8.54e-3 * 70.7012 * A[2] / A[0]
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
X = infer_domain(f, gts, [])
prim = sig_constrained_relaxation(f, main_gts, [], X, 'P', p=0, q=1, ell=0)
dual = sig_constrained_relaxation(f, main_gts, [], X, 'D', p=0, q=1, ell=0)

# Solve SAGE relaxations, and print the resulting objective values.
prim.solve(verbose=False)
dual.solve(verbose=False)
print('\n')
print(prim.value)
print(dual.value)

# Solution recovery
solns = sig_solrec(dual)
print(f(solns[0]))

#
#   It was possible to produce a tight bound on the equality-constrained problem all along.
#

x = standard_sig_monomials(4)
A = x[:3]
P = x[3]
f = 1e4 * sum(A)
main_gts = [
    1e4 + 1e-2 * A[2] / A[0] - 7.0711 / A[0],
    1e4 + 8.54e-3 * P/ A[0] - 6.0385e-1 * (1.0 / A[0] + 1.0 / A[1])
]
bounds = [
    1e4 - 1e4 * A[0], 1e4 * A[0] - 1e-4,
    1e4 - 1e4 * A[1], 1e4 * A[1] - 7.0711,
    1e4 - 1e4 * A[2], 1e4 * A[2] - 1e-4,
    1e4 - 1e4 * P, 1e4 * P - 1e-4
]
gts = main_gts + bounds
eqs = [70.7107 / A[0] + P / A[0] - P / A[2]]
X = infer_domain(f, bounds, [])
prim = sig_constrained_relaxation(f, main_gts, eqs, X, 'P', p=0, q=1, ell=0)
dual = sig_constrained_relaxation(f, main_gts, eqs, X, 'D', p=0, q=1, ell=0)
# Solve SAGE relaxations, and print the resulting objective values.
prim.solve(verbose=False)
dual.solve(verbose=False)
print('\n')
print(prim.value)
print(dual.value)
