from sageopt.coniclifts.constraints.set_membership.setmem import SetMembership
from sageopt.coniclifts.constraints.set_membership.sage_cone import ExpCoverHelper
from sageopt.coniclifts.base import Variable, Expression


class PrimalSagePolyCone(SetMembership):

    def __init__(self, c, alpha, name=None, expcovers=None):
        if name is None:
            name = ''
        self.name = name
        self.alpha = alpha
        self.m = alpha.shape[0]
        self.c = Expression(c)  # self.c is now definitely an ndarray of ScalarExpressions.
        self.ech = ExpCoverHelper(self.alpha, self.c, expcovers)
        self._variables = self.c.variables()
        self._initialize_primary_variables()
        pass

    def _initialize_primary_variables(self):
        raise NotImplementedError()

    def variables(self):
        return self._variables

    def conic_form(self):
        raise NotImplementedError()


class DualSagePolyCone(SetMembership):

    def __init__(self, v, alpha, c=None, name=None, expcovers=None):
        if c is None:
            self.c = Variable(shape=(alpha.shape[0],), name='dummy').view(Expression)
        else:
            self.c = Expression(c)
        self.alpha = alpha
        self.ech = ExpCoverHelper(self.alpha, self.c, expcovers)
        self.m = alpha.shape[0]
        self.n = alpha.shape[1]
        self.v = v
        self.name = name
        self.mu_vars = dict()
        self._variables = self.v.variables()
        self._initialize_primary_variables()
        pass

    def variables(self):
        return self._variables

    def conic_form(self):
        raise NotImplementedError()

    def _initialize_primary_variables(self):
        raise NotImplementedError()


