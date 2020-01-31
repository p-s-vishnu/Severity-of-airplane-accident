from sklearn.preprocessing import StandardScaler
import numpy as np

"""
Apply Standard scalar
"""


def standardize_all(X_train=None, X_test=None):

    ss = StandardScaler().fit(X_train)
    return (ss.transform(X_train),
            ss.transform(X_test))


"""
Apply log transformation
"""


def log_transform(X):
    return np.log1p(X)


def power_transform(X, n=2, columns=None):
    if columns:
        return np.power(X[columns], n)
    return np.power(X, n)
