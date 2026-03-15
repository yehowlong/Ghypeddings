from sklearn.cluster import AgglomerativeClustering
from clusterers.utils import calculate_metrics

def agglomerative_clustering(X,y,n_clusters =2, linkage = 'ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage)
    labels = model.fit_predict(X)
    return calculate_metrics(y,labels)