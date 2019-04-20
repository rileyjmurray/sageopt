

class Constraint(object):

    def is_affine(self):
        raise NotImplementedError()

    def is_elementwise(self):
        raise NotImplementedError()

    def is_setmem(self):
        raise NotImplementedError()

    def variables(self):
        raise NotImplementedError()

    def conic_form(self):
        """
        :return: A_vals, A_rows, A_cols, b, K, sep_K
            A_vals - list (of floats)
            A_rows - numpy 1darray (of integers)
            A_cols - list (of integers)
            b - numpy 1darray (of floats)
            K - list (of coniclifts Cone objects)
            sep_K - list (of appropriately annotated coniclifts Cone objects)
        """
        raise NotImplementedError()

    def violation(self):
        raise NotImplementedError()
