from sigmoid import sigmoid
import numpy as np

def sigmoidGradient(z):
    g_prime = sigmoid(z)*(1-sigmoid(z))
    return g_prime