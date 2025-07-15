from sklearn.cluster import DBSCAN

def cluster_embeddings(embeddings):
    cluster = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
    clusters = cluster.fit_predict(embeddings)
    return clusters