

class Solver(object):

    @staticmethod
    def apply(c, A, b, K, sep_K, destructive, compilation_options):
        raise NotImplementedError()

    @staticmethod
    def solve_via_data(data, params):
        raise NotImplementedError()

    @staticmethod
    def parse_result(solver_output, inv_data, var_mapping):
        raise NotImplementedError()

    @staticmethod
    def is_installed():
        raise NotImplementedError()
