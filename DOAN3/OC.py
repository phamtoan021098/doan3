from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
n_points_per_cluster = 250
Cluster1 = [-4, -3] + .8 * np.random.randn(n_points_per_cluster, 2)
Cluster2 = [3, -1.5] + .1 * np.random.randn(n_points_per_cluster, 2)
Cluster3 = [2, -1] + .2 * np.random.randn(n_points_per_cluster, 2)
Cluster4 = [-4, 5] + .3 * np.random.randn(n_points_per_cluster, 2)
Cluster5 = [0,-0.5] + 1.2 * np.random.randn(n_points_per_cluster, 2)
Cluster6 = [4, 7] + 2 * np.random.randn(n_points_per_cluster, 2)
X = np.vstack((Cluster1,Cluster2,Cluster3,Cluster4,Cluster5, Cluster6))
print(X)
clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)

clust.fit(X)
labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=0.5)
labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=2)
space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = X[clust.labels_ == klass]
    plt.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
plt.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)

plt.show()