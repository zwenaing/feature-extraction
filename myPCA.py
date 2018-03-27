import numpy as np
from sklearn.preprocessing import scale


class myPCA:

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        """
        Find the principal components of the data.
        :param X: the input data.
        """
        n, d = X.shape
        scaled_X = scale(X, axis=0, with_std=False) # Perform zero mean for each feature
        covariance = np.dot(np.transpose(scaled_X), scaled_X) / n   # calculate covariance matrix
        U, S, V = np.linalg.svd(covariance)
        self.principal_components = U[:, :self.n_components]

    def transform(self, X):
        """
        Transform the given data X into a lower dimensional space.
        :param X: the input data X.
        :return: the transformed data.
        """
        scaled_X = scale(X, axis=0, with_std=False)
        return np.dot(scaled_X, self.principal_components)