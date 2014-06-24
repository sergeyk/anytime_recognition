""" the stochastic gradient descent solver
"""

import numpy as np

class SGDProblem(object):
    def __init__(self):
        pass
    
    def x_init(self):
        """Returns the initial value of x
        """
        raise NotImplementedError

    def func_and_grad(self, x, batchsize = 1):
        """Returns the function value and the gradient of a randomly sampled
        batch.
        Input:
            x: the current variable values
            batchsize: the size of the batch, default 1.
        Output:
            f: the function value
            g: the gradient as a vector of the same size as x
        """
        raise NotImplementedError
    
    def callback(self, x, iter_count):
        """ An optional callback function that is called after each sgd update.
        It is designed to return the updated parameters in case you need to do
        post processing like normalization.
        Input:
            x: the current variable values (after this iteration's update)
            iter_count: the count of the current iteration
        Output:
            x_out: the postprocessed param.
        """
        return x

class SGDOptimizer(object):
    """The general SGD optimizer that MINIMIZES an object function.
    """
    def __init__(self, problem, lr, t0, batchsize = 1, adagrad = False):
        """Initializes a sgd optimizer. 
        Input:
            problem: a SGDProblem instance
            lr: the initial learning rate
            t0: the steps after which we do annealing. the learning rate of step
                t is lr * min(1, t0 / (t+1))
            batchsize: the batch size for each iteration. Default 1.
            adagrad: if True, use adagrad to determine the learning rate.
        """
        self._problem = problem
        self._lr = lr
        self._t0 = t0
        self._batchsize = batchsize
        self._adagrad = adagrad
        # the initial parameter
        self._x = self._problem.x_init()
        self._t = 0
    
    def run(self, maxiter = 1000):
        """Run (or continue, if you have already run it before) the sgd 
        optimizer for maxiter steps
        """
        for _ in range(maxiter):
            f, g = self._problem.func_and_grad(self._x, self._batchsize)
            
            # set up the learning rate
            if self._adagrad:
                raise NotImplementedError
            else:
                current_lr = self._lr * min(1., self._t0 / (self._t + 1.))
            
            self._x -= current_lr * g
            self._t += 1
            # run callback
            self._problem.callback(self._x, self._t)
        return self._x
    

class ToyLinearRegressionProblem(SGDProblem):
    """ToyLinearRegressionProblem implements a toy regression example from
    R^2 to R, represented by x^T a = b, where a is randomly sampled from 
    [-10,10] every time. w is the parameter we want to solve for.
    """
    def __init__(self, noise):
        """init randomly generates a ground truth weight.
        """
        self._weight = np.random.randn(2)
        self._noise = noise
    
    def x_init(self):
        """We always start from 0
        """
        return np.zeros(2)

    def func_and_grad(self, x, batchsize = 1):
        """Generate ground truth and then computes the func and gradients
        """
        # ground truth with noise
        A = np.random.rand(batchsize, 2) * 20. - 10.
        b = np.dot(A, self._weight) + np.random.randn(batchsize)
        # prediction
        pred = np.dot(A, x)
        diff = b - pred
        f = np.dot(diff, diff)
        g = - np.dot(diff, A)
        g *= 2.
        return f, g
    
    def callback(self, x, iter_count):
        print x, iter_count
    
def demo():
    problem = ToyLinearRegressionProblem(0.1)
    optimizer = SGDOptimizer(problem, 0.001, 1, 64)
    optimizer.run(100000)
    print problem._weight

if __name__ == "__main__":
    demo()