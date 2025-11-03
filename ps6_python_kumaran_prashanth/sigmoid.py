import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def sigmoid(z):
    g = (1/(1+np.exp(-z)))
    return g
