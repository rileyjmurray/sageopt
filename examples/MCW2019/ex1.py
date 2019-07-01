from sageopt import conditional_sage_data, sig_dual, standard_sig_monomials, sig_solrec

n = 3
y = standard_sig_monomials(n)
f = 0.5 * y[0] * y[1] ** -1 - y[0] - 5 * y[1] ** -1
gts = [100 -  y[1] * y[2] ** -1 - y[1] - 0.05 * y[0] * y[2],
       y[0] - 70,
       y[1] - 1,
       y[2] - 0.5,
       150 - y[0],
       30 - y[1],
       21 - y[2]]
eqs = []
X = conditional_sage_data(f, gts, eqs)
dual = sig_dual(f, ell=0, X=X)
dual.solve(verbose=False)
solutions = sig_solrec(dual)
best_soln = solutions[0]
print('The level 0 SAGE bound is ... ')
print('\t' + str(dual.value))
print("The recovered solution has objective value ...")
print('\t' + str(f(best_soln)))
print("The recovered solution has constraint violation ...")
constraint_levels = min([g(best_soln) for g in gts])
violation = 0 if constraint_levels >= 0 else -constraint_levels
print('\t' + str(violation))
dual = sig_dual(f, ell=3, X=X)
dual.solve(verbose=False)
print('The level 3 SAGE bound is ... ')
print('\t' + str(dual.value))

