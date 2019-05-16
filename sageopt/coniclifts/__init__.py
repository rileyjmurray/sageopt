from sageopt.coniclifts.base import Variable, ScalarVariable, Expression
from sageopt.coniclifts.operators.affine import *
from sageopt.coniclifts.operators.relent import relent
from sageopt.coniclifts.operators.exp import weighted_sum_exp
from sageopt.coniclifts.operators.norms import vector2norm
from sageopt.coniclifts.compilers import compile_constrained_system, compile_linear_expression, compile_problem
from sageopt.coniclifts.constraints.set_membership.product_cone import PrimalProductCone
from sageopt.coniclifts.constraints.set_membership.sage_cone import PrimalSageCone, DualSageCone
from sageopt.coniclifts.constraints.set_membership.conditional_sage_cone import PrimalCondSageCone, DualCondSageCone
from sageopt.coniclifts.problems.problem import Problem
from sageopt.coniclifts.problems.solvers.mosek import Mosek
from sageopt.coniclifts.problems.solvers.ecos import ECOS
from sageopt.coniclifts.standards.constants import minimize as MIN
from sageopt.coniclifts.standards.constants import maximize as MAX
from sageopt.coniclifts.standards.constants import solved as SOLVED
from sageopt.coniclifts.standards.constants import inaccurate as INACCURATE


def clear_variable_indices():
    ScalarVariable._SCALAR_VARIABLE_COUNTER = 0
    # noinspection PyProtectedMember
    Variable._VARIABLE_GENERATION += 1

