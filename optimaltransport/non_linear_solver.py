import numpy as np


def LagrangeSolver(X, T, m_r):
    GWidth = np.sum([np.sqrt(X[i]) for i in range(len(X))])

    p_r = np.array([np.sqrt(X[i]) / GWidth for i in range(len(X))])

    fun = np.sum([X[i] / (T[i] + p_r[i] * m_r) for i in range(len(X))])
    ret = np.ceil(np.array(p_r * m_r))
    return ret, fun
