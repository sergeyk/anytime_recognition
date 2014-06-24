"""This script implements the tree-based regularization term as proposed in
the paper:
    Ruslan Salakhutdinov, Antonio Torralba, Josh Tenenbaum.
    Learning to Share Visual Appearance for Multiclass Object Detection.
    CVPR 2011.
We do not implement the tree structure learning (CRP) part and assume the 
tree to be known.
"""

from iceberk import classifier, mpi, mathutil
import numpy as np


class SolverTreeReg(classifier.Solver):
    '''SolverTreeReg, Documentation to follow.
    
    Regargs should contain the 0-1 matrix of size
        [num_hidden_nodes x num_classes]
    '''
    def presolve(self, X, Y, weight, param_init):
        self._X = X.reshape((X.shape[0],np.prod(X.shape[1:])))
        if len(Y.shape) == 1:
            self._K = mpi.COMM.allreduce(Y.max(), op=max) + 1
        else:
            # We treat Y as a two-dimensional matrix
            Y = Y.reshape((Y.shape[0],np.prod(Y.shape[1:])))
            self._K = Y.shape[1]
        self._Y = Y
        self._weight = weight
        # compute the number of data
        if weight is None:
            self._num_data = mpi.COMM.allreduce(X.shape[0])
        else:
            self._num_data = mpi.COMM.allreduce(weight.sum())
        self._dim = self._X.shape[1]
        self._pred = np.empty((X.shape[0], self._K), dtype = X.dtype)
        tree = self._regargs['tree']
        self._Khidden = tree.shape[0]
        if param_init is None:
            param_init = np.zeros(self._Khidden * (self._dim) + self._K)
        elif len(param_init) == 2:
            # the initialization is w and b
            param_init = np.hstack((param_init[0].flatten(), 
                                    param_init[1].flatten()))
        # gradient cache
        self._glocal = np.empty(param_init.shape)
        self._g = np.empty(param_init.shape)
        # just to make sure every node is on the same page
        mpi.COMM.Bcast(param_init)
        return param_init
    
    def postsolve(self, lbfgs_result):
        wb = lbfgs_result[0]
        Khidden = self._Khidden
        w = wb[: Khidden * self._dim].reshape(self._dim, Khidden).copy()
        b = wb[Khidden * self._dim :].copy()
        return w, b
    
    @staticmethod
    def obj(wb,solver):
        '''
        The objective function used by fmin
        '''
        # obtain w and b
        Khidden = solver._Khidden
        dim = solver._dim
        whidden = wb[:Khidden*dim].reshape((dim, Khidden))
        tree = solver._regargs['tree']
        w = mathutil.dot(whidden, tree)
        b = wb[Khidden*dim:]
        # pred is a matrix of size [num_datalocal, K]
        mathutil.dot(solver._X, w, out = solver._pred)
        solver._pred += b
        # compute the loss function
        flocal,gpred = solver.loss(solver._Y, solver._pred, solver._weight,
                                   **solver._lossargs)
        mathutil.dot(mathutil.dot(solver._X.T, gpred), tree.T,
                out = solver._glocal[:Khidden*dim].reshape(dim, Khidden))
        solver._glocal[Khidden*dim:] = gpred.sum(axis=0)
        
        # add regularization term, but keep in mind that we have multiple nodes
        freg, greg = solver.reg(whidden, **solver._regargs)
        flocal += solver._num_data * solver._gamma * freg / mpi.SIZE
        solver._glocal[:Khidden*dim] += solver._num_data * solver._gamma \
                * greg.ravel() / mpi.SIZE
        # do mpi reduction
        mpi.barrier()
        f = mpi.COMM.allreduce(flocal)
        mpi.COMM.Allreduce(solver._glocal, solver._g)
        return f, solver._g
