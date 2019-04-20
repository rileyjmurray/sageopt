from sageopt.coniclifts.constraints.constraint import Constraint


class SetMembership(Constraint):

    def is_affine(self):
        return True

    def is_setmem(self):
        return True

    def is_elementwise(self):
        return False

    def variables(self):
        raise NotImplementedError()

    def conic_form(self):
        raise NotImplementedError()
