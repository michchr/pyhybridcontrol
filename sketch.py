from models.micro_grid_devices import DewhModelGenerator, Dewh
from models.parameters import dewh_p
from models.mld_model import MldModel

import scipy.sparse as scs
import scipy.linalg as scl
import numpy.linalg as npl
import numpy as np

mld = MldModel(A=np.reshape([1, 2, 3, 4], (2, 2)),
               B1=np.atleast_2d([4, 5]).T,
               B2=np.atleast_2d([11, 12]).T,
               B3=np.atleast_2d([6, 7]).T,
               B4=np.atleast_2d([8, 9]).T,
               b5=np.ones((2, 1)), d5=np.ones((2, 1)), g6=np.ones((2, 1)))
