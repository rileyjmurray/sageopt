import sageopt as so
import numpy as np
from sageopt.interop.gpkit import gpkit_model_to_sageopt_model
from gpkit import Variable, Model, SignomialsEnabled
from gpkit.constraints.sigeq import SingleSignomialEquality
#
# Build GPKit model
#
x = Variable('x')
y = Variable('y')
with SignomialsEnabled():
    constraints = [0.2 <= x, x <= 0.95, SingleSignomialEquality(x + y, 1)]
gpkm = Model(x*y, constraints)
#
#   Recover data for the sageopt model
#
som = gpkit_model_to_sageopt_model(gpkm)  # a dict
sp_eqs, gp_gts = som['sp_eqs'], som['gp_gts']
f = som['f']
X = so.infer_domain(f, gp_gts, [])
prob = so.sig_constrained_relaxation(f, gp_gts, sp_eqs, X,  p=1)
#
#   Solve and check solution
#
prob.solve(solver='ECOS', verbose=False)
soln = so.sig_solrec(prob)[0]
geo_soln = np.exp(soln)
vkmap = som['vkmap']
x_val = geo_soln[vkmap[x.key]]
y_val = geo_soln[vkmap[y.key]]
