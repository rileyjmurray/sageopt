from sageopt.coniclifts.base import Variable, ScalarVariable, Expression
from sageopt.coniclifts.operators.affine import *
from sageopt.coniclifts.operators.relent import relent
from sageopt.coniclifts.operators.exp import weighted_sum_exp
from sageopt.coniclifts.operators.norms import vector2norm
from sageopt.coniclifts.compilers import compile_constrained_system, compile_problem
from sageopt.coniclifts.constraints.constraint import Constraint
from sageopt.coniclifts.constraints.set_membership.product_cone import PrimalProductCone
from sageopt.coniclifts.constraints.set_membership.sage_cones import PrimalSageCone, DualSageCone
from sageopt.coniclifts.problems.problem import Problem
from sageopt.coniclifts.problems.solvers.mosek import Mosek
from sageopt.coniclifts.problems.solvers.ecos import ECOS
from sageopt.coniclifts.standards.constants import minimize as MIN
from sageopt.coniclifts.standards.constants import maximize as MAX
from sageopt.coniclifts.standards.constants import solved as SOLVED
from sageopt.coniclifts.standards.constants import inaccurate as INACCURATE
from sageopt.coniclifts.standards.constants import failed as FAILED
from sageopt.coniclifts.cones import Cone


def clear_variable_indices():
    """
    Reset coniclifts' internal counter for ScalarVariable objects (to zero), and
    increment coniclifts' internal counter for the "generation" of Variable objects.

    Do not call this function while building a model with coniclifts. Only
    use it when you are done working with all Variable objects which have been
    previously declared.

    Strictly speaking, this function only needs to be called when more than
    2**62 scalar variables have been declared in the same python
    environment. The only way this could happen is if coniclifts were used in a
    loop for a very, very, very long time. Coniclifts' backend will automatically
    raise an error if this function wasn't called frequently enough.
    """
    ScalarVariable._SCALAR_VARIABLE_COUNTER = 0
    # noinspection PyProtectedMember
    Variable._VARIABLE_GENERATION += 1
    pass


def presolve_trivial_age_cones(true_or_false=False):
    """
    Set coniclifts' behavior for SAGE constraints.

    If ``true_or_false=True``, then coniclifts will solve a series of small optimization
    problems whenever a SAGE constraint (primal or dual) is declared. This presolve
    behavior reduces the size of the final SAGE relaxation which needs to be solved,
    and in some sense improves problem conditioning.

    The default value for ``true_or_false`` in this function's signature represents
    sageopt's default behavior for this setting.
    """
    import sageopt.coniclifts.constraints.set_membership.sage_cones as sc
    sc.SETTINGS['presolve_trivial_age_cones'] = true_or_false


def heuristic_reduce_cond_age_cones(true_or_false=True):
    """
    Set coniclifts' behavior for conditional SAGE constraints.

    If ``true_or_false=True``, then coniclifts will take a particular reduction that
    is without-loss-of-generality for ordinary SAGE constraints, and apply that
    reduction to conditional SAGE constraints.

    The default value for ``true_or_false`` in this function's signature represents
    sageopt's default behavior for this setting.
    """
    import sageopt.coniclifts.constraints.set_membership.sage_cones as sc
    sc.SETTINGS['heuristic_reduction'] = true_or_false


def age_cone_reduction_solver(solver_str='ECOS'):
    """
    Use the provided string as the solver argument when any optimization-based
    presolve is employed in SAGE constraints.

    The default value for ``solver_str`` in this function's signature represents
    sageopt's default behavior for this setting.
    """
    import sageopt.coniclifts.constraints.set_membership.sage_cones as sc
    sc.SETTINGS['reduction_solver'] = solver_str


def sum_age_force_equality(true_or_false=False):
    """
    Set coniclifts' behavior for compiling PrimalSageCone constraints.

    If ``true_or_false=True``, then a PrimalSageCone constraint ``con`` will
    tell the solver to require the values of ``con.age_vectors`` sum to ``con.c``.

    If ``true_or_false=False``, then a PrimalSageCone constraint ``con`` will
    tell the solver to require the sum of values of ``con.age_vectors`` is <= ``con.c``.

    The default value for ``true_or_false`` in this function's signature represents
    sageopt's default behavior for this setting.
    """
    import sageopt.coniclifts.constraints.set_membership.sage_cones as sc
    sc.SETTINGS['sum_age_force_equality'] = true_or_false


def compact_sage_duals(true_or_false=True):
    """
    Decide how coniclifts compiles constraints
        ``v[i] * log(v[i] / v[j]) <= (alpha[i,:] - alpha[j,:]) @ mu_i``    (*)
    which appear in DualSageCone objects.

    If ``true_or_false=True``, then (*) compiles into a constraint that maps
    ``(v[i],v[j],mu_i)`` into a single exponential cone.

    If ``true_or_false=False``, then compiling (*) introduces an epigraph
    variable ``epi`` plus the constraints ``v[i] * log(v[i]/v[j]) <= epi``
    and ``epi <= (alpha[i,:] - alpha[j,:]) @ mu_i``.

    The default value for ``true_or_false`` in this function's signature represents
    sageopt's default behavior for this setting.
    """
    import sageopt.coniclifts.constraints.set_membership.sage_cones as sc
    sc.SETTINGS['compact_dual'] = true_or_false


def kernel_basis_age_witnesses(true_or_false=True):
    """
    Set coniclifts' behavior for SAGE constraints.

    If ``true_or_false=True``, then coniclifts will represent primal ordinary
    SAGE constraints using no equality constraints within an AGE cone. This comes
    at the expense of taking a longer time for coniclifts to setup the problem.
    It also might affect solver behavior (probably for the better).

    The default value for ``true_or_false`` in this function's signature represents
    sageopt's default behavior for this setting.
    """
    import sageopt.coniclifts.constraints.set_membership.sage_cones as sc
    sc.SETTINGS['kernel_basis'] = true_or_false
