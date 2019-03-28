import cvxpy as cvx

v = cvx.Variable(50000, nonneg=True)
p = cvx.Problem(cvx.Minimize(cvx.sum(v)))
p.solve()

import re as regex

pat = regex.compile('([0-9]+)')

pat.match()