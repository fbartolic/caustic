# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["CeleriteBaseOp"]

import os
import theano
from theano import gof
import theano.tensor as tt


class CeleriteBaseOp(gof.COp):

    __props__ = ("J", "n_rhs")
    num_input = 0
    output_ndim = ()
    func_file = None
    func_name = None

    def __init__(self, J=-1, n_rhs=-1):
        self.J = int(J)
        self.n_rhs = int(n_rhs)
        super(CeleriteBaseOp, self).__init__(self.func_file, self.func_name)

    # def c_code_cache_version(self):
    #     return (0, 0, 1)

    def c_headers(self, compiler):
        return ["theano_helpers.h"]

    def c_header_dirs(self, compiler):
        return [
            os.path.dirname(os.path.abspath(__file__)),
            "/usr/local/include/eigen3"
        ]

    def c_compile_args(self, compiler):
        args = ["-march=native", "-O2", "-DNDEBUG"]
        if 0 < self.J <= 32:
            args.append("-DCELERITE_J={0}".format(self.J))
            args.append("-DCELERITE_J2={0}".format(self.J * self.J))
            if self.J == 1:
                args.append("-DCELERITE_J_ORDER=Eigen::ColMajor")
            else:
                args.append("-DCELERITE_J_ORDER=Eigen::RowMajor")
        if 0 < self.n_rhs <= 32:
            args.append("-DCELERITE_NRHS={0}".format(self.n_rhs))
            args.append("-DCELERITE_JNRHS={0}".format(self.n_rhs * self.J))
            if self.n_rhs == 1:
                args.append("-DCELERITE_NRHS_ORDER=Eigen::ColMajor")
            else:
                args.append("-DCELERITE_NRHS_ORDER=Eigen::RowMajor")
            if self.n_rhs == 1 and self.J == 1:
                args.append("-DCELERITE_JNRHS_ORDER=Eigen::ColMajor")
            else:
                args.append("-DCELERITE_JNRHS_ORDER=Eigen::RowMajor")
        return args

    def make_node(self, *args):
        if len(args) != self.num_input:
            raise ValueError("expected {0} inputs".format(self.num_input))
        dtype = theano.config.floatX
        in_args = []
        for a in args:
            try:
                a = tt.as_tensor_variable(a)
            except tt.AsTensorError:
                pass
            else:
                dtype = theano.scalar.upcast(dtype, a.dtype)
            in_args.append(a)
        out_args = [
            tt.TensorType(dtype=dtype, broadcastable=[False] * ndim)()
            for ndim in self.output_ndim]
        return gof.Apply(self, in_args, out_args)
