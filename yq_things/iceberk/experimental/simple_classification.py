from iceberk import classifier
import numpy as np

X = np.random.rand(100,2)
X = np.vstack((X + [1, 1],
               X + [1, -1],
               X + [-1, 1],
               X + [-1, -1]))
Y = np.tile(np.arange(4),(100, 1)).T.flatten()
Y = classifier.to_one_of_k_coding(Y, fill = 0)

solver = classifier.SolverMC(0.01, 
                             classifier.Loss.loss_multiclass_logistic,
                             classifier.Reg.reg_l2)

w, b = solver.solve(X, Y)