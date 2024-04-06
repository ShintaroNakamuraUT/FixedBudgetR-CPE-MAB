import numpy as np
from scipy.optimize import minimize


def NonLPSolver(X, T, m_r):
    dimensions = len(X)
    bnds = ((0, 1),) * dimensions

    def objective(x):
        ret = np.sum(
            [X[i] / (T[i] + np.ceil(x[i] * m_r)) for i in range(np.int(dimensions))]
        )
        return ret

    def constraint(x):
        ret = np.sum(x) - 1
        return ret

    cons = {"type": "eq", "fun": constraint}
    cons = [cons]
    sol = minimize(objective, X, method="SLSQP", bounds=bnds, constraints=cons)

    ret = np.ceil(np.array(sol.x) * m_r)
    fun = sol.fun
    return ret, fun


def LagrangeSolver(X, T, m_r):
    GWidth = np.sum([np.sqrt(X[i]) for i in range(len(X))])

    p_r = np.array([np.sqrt(X[i]) / GWidth for i in range(len(X))])

    fun = np.sum([X[i] / (T[i] + p_r[i] * m_r) for i in range(len(X))])
    ret = np.ceil(np.array(p_r * m_r))
    return ret, fun
