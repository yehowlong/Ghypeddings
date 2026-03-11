from sklearn.mixture import GaussianMixture
from Ghypeddings.clusterers.utils import calculate_metrics

def gaussian_mixture(X,y,n_components=2):
    model = GaussianMixture(n_components=2)
    y_pred = model.fit_predict(X)
    return calculate_metrics(y,y_pred)