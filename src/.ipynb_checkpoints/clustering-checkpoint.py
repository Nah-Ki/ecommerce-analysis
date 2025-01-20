from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

def perform_clustering(data, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    db_index = davies_bouldin_score(data, clusters)
    return clusters, db_index
