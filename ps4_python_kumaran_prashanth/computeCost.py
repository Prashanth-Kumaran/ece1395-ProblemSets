import numpy as np
import matplotlib.pyplot as plt
def computeCost(X, y, theta):
    m = y.shape[0]
    J = (1/(2*m))*((y-(X@theta)).T)@(y-(X@theta))
    return float(J)