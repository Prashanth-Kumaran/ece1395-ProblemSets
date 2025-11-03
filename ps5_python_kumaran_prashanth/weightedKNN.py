import numpy as np
from scipy.spatial.distance import cdist

def weightedKNN(X_train, y_train, X_test, sigma):
    classes = np.unique(y_train)
    distances = cdist(X_test, X_train, 'euclidean')
    weights = np.exp(-distances**2 / (sigma**2))
    d = X_test.shape[0]
    y_predict = np.zeros(d)
    
    for i in range(d):
        w_i = weights[i, :] 
        weighted_votes = {}

        for c in classes:
            mask = (y_train == c).flatten()
            weighted_votes[c] = np.sum(w_i[mask])

        y_predict[i] = max(weighted_votes, key=weighted_votes.get) 

    return y_predict