from clusterers.utils import calculate_metrics
import skfuzzy as fuzz
import numpy as np

def fuzzy_c_mean(X,y,n_clusters=5,power=2,error=0.005,maxiter=1000,init=None):
    X_transposed = np.transpose(X)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_transposed, n_clusters, power, error=error, maxiter=maxiter, init=init)
    y_pred = np.argmax(u, axis=0)
    return calculate_metrics(y,y_pred)