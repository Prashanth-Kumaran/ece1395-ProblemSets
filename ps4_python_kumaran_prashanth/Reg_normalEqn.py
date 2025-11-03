import numpy as np
def Reg_normalEqn(X_train, y_train, lamb):
    D = np.eye(X_train.shape[1]) 
    D[0][0] = 0
    theta = np.linalg.pinv(X_train.T@X_train + lamb*D)@X_train.T@y_train
    return theta
