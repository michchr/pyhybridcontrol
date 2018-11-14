from models.agents import DewhModelGenerator, Dewh
from models.parameters import dewh_p
from models.mld_model import MldModel

import scipy.sparse as scs
import scipy.linalg as scl
import numpy.linalg as npl
import numpy as np

mld = MldModel(A=np.reshape([1, 2, 3, 4], (2, 2)),
               B1=np.atleast_2d([[4, 5],[6,7]]),
               B2=np.atleast_2d([[11, 12],[9,10]]),
               B3=np.atleast_2d([[6, 7],[10,2]]),
               B4=np.atleast_2d([[8, 9],[102,2]]),
               b5=np.ones((2, 1)), E1=np.ones((2,2)), g6=np.ones((2,1)),
               dt = 1)

u = np.random.rand(5,2)
delta = np.random.randint(0,2,size=(5,2))
z = np.random.rand(5,2)
omega = np.random.rand(5,2)

mld.lsim(u, delta, z, omega)