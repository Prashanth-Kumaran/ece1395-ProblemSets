import numpy as np
from sklearn.linear_model import LogisticRegression

def logReg_multi(X_train, y_train, X_test):
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_test = X_test.shape[0]

    probs = np.zeros((n_test, n_classes))
    for idx, c in enumerate(classes):
        y_c = np.where(y_train == c, 1, 0)
        mdl_c = LogisticRegression(random_state=0).fit(X_train, y_c)
        probs[:, idx] = mdl_c.predict_proba(X_test)[:, 1]


    y_predict = classes[np.argmax(probs, axis=1)]
    return y_predict.reshape(-1, 1)