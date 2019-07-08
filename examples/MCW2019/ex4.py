from sageopt import standard_sig_monomials, local_refine_polys_from_sigs
from sageopt import sig_constrained_dual, sig_solrec

x = standard_sig_monomials(6)
f = x[0]**6 - x[1]**6 + x[2]**6 - x[3]**6 + x[4]**6 - x[5]**6 + x[0] - x[1]

expr1 = 2*x[0]**6 + 3*x[1]**2 + 2*x[0]*x[1] + 2*x[2]**6 + 3*x[3]**2 + 2*x[2]*x[3] + 2*x[4]**6 + 3*x[5]**2 + 2*x[4]*x[5]
expr2 = 2*x[0]**2 + 5*x[1]**2 + 3*x[0]*x[1] + 2*x[2]**2 + 5*x[3]**2 + 3*x[2]*x[3] + 2*x[4]**2 + 5*x[5]**2 + 3*x[4]*x[5]
expr3 = 3*x[0]**2 + 2*x[1]**2 - 4*x[0]*x[1] + 3*x[2]**2 + 2*x[3]**2 - 4*x[2]*x[3] + 3*x[4]**2 + 2*x[5]**2 - 4*x[4]*x[5]

expr4 = x[0]**2 + 6*x[1]**2 - 4*x[0]*x[1] + x[2]**2 + 6*x[3]**2 - 4*x[2]*x[3] + x[4]**2 + 6*x[5]**2 - 4*x[4]*x[5]
expr5 = x[0]**2 + 6*x[1]**2 - 4*x[0]*x[1] + x[2]**2 + 6*x[3]**2 - 4*x[2]*x[3] + x[4]**2 + 6*x[5]**2 - 4*x[4]*x[5]

gts = [
    expr3,
    expr4,
    expr5,
    1 - expr1,
    1 - expr2,
    1 - expr3,
    1 - expr4,
    1 - expr5
]
eqs = []

dual = sig_constrained_dual(f, gts, eqs, 1, 1, 0)
dual.solve(verbose=False, solver='MOSEK')  # ECOS fails
solns = sig_solrec(dual)
x0 = solns[0]
x_star = local_refine_polys_from_sigs(f, gts, eqs, x0)

print()
print(dual.value)
f_poly = f.as_polynomial()
print(f_poly(x_star))
print(x_star)
