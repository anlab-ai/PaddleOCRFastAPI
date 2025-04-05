import warnings
from numbers import Integral, Real

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, ClusterMixin, _fit_context
from sklearn.metrics.pairwise import _VALID_METRICS
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import _check_sample_weight
from sklearn.cluster._dbscan_inner import dbscan_inner
from shapely import Polygon
import matplotlib.pyplot as plt
import cv2


@validate_params(
	{
		"X": ["array-like", "sparse matrix"],
		"sample_weight": ["array-like", None],
	},
	prefer_skip_nested_validation=False,
)


def dbscan(
	X,
	eps=0.5,
	*,
	min_samples=5,
	metric="minkowski",
	metric_params=None,
	algorithm="auto",
	leaf_size=30,
	p=2,
	sample_weight=None,
	n_jobs=None,
):
	"""Perform DBSCAN clustering from vector array or distance matrix.

	Read more in the :ref:`User Guide <dbscan>`.

	Parameters
	----------
	X : {array-like, sparse (CSR) matrix} of shape (n_samples, n_features) or \
			(n_samples, n_samples)
		A feature array, or array of distances between samples if
		``metric='precomputed'``.

	eps : float, default=0.5
		The maximum distance between two samples for one to be considered
		as in the neighborhood of the other. This is not a maximum bound
		on the distances of points within a cluster. This is the most
		important DBSCAN parameter to choose appropriately for your data set
		and distance function.

	min_samples : int, default=5
		The number of samples (or total weight) in a neighborhood for a point
		to be considered as a core point. This includes the point itself.

	metric : str or callable, default='minkowski'
		The metric to use when calculating distance between instances in a
		feature array. If metric is a string or callable, it must be one of
		the options allowed by :func:`sklearn.metrics.pairwise_distances` for
		its metric parameter.
		If metric is "precomputed", X is assumed to be a distance matrix and
		must be square during fit.
		X may be a :term:`sparse graph <sparse graph>`,
		in which case only "nonzero" elements may be considered neighbors.

	metric_params : dict, default=None
		Additional keyword arguments for the metric function.

		.. versionadded:: 0.19

	algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
		The algorithm to be used by the NearestNeighbors module
		to compute pointwise distances and find nearest neighbors.
		See NearestNeighbors module documentation for details.

	leaf_size : int, default=30
		Leaf size passed to BallTree or cKDTree. This can affect the speed
		of the construction and query, as well as the memory required
		to store the tree. The optimal value depends
		on the nature of the problem.

	p : float, default=2
		The power of the Minkowski metric to be used to calculate distance
		between points.

	sample_weight : array-like of shape (n_samples,), default=None
		Weight of each sample, such that a sample with a weight of at least
		``min_samples`` is by itself a core sample; a sample with negative
		weight may inhibit its eps-neighbor from being core.
		Note that weights are absolute, and default to 1.

	n_jobs : int, default=None
		The number of parallel jobs to run for neighbors search. ``None`` means
		1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
		using all processors. See :term:`Glossary <n_jobs>` for more details.
		If precomputed distance are used, parallel execution is not available
		and thus n_jobs will have no effect.

	Returns
	-------
	core_samples : ndarray of shape (n_core_samples,)
		Indices of core samples.

	labels : ndarray of shape (n_samples,)
		Cluster labels for each point.  Noisy samples are given the label -1.

	See Also
	--------
	DBSCAN : An estimator interface for this clustering algorithm.
	OPTICS : A similar estimator interface clustering at multiple values of
		eps. Our implementation is optimized for memory usage.

	Notes
	-----
	For an example, see :ref:`sphx_glr_auto_examples_cluster_plot_dbscan.py`.

	This implementation bulk-computes all neighborhood queries, which increases
	the memory complexity to O(n.d) where d is the average number of neighbors,
	while original DBSCAN had memory complexity O(n). It may attract a higher
	memory complexity when querying these nearest neighborhoods, depending
	on the ``algorithm``.

	One way to avoid the query complexity is to pre-compute sparse
	neighborhoods in chunks using
	:func:`NearestNeighbors.radius_neighbors_graph
	<sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
	``mode='distance'``, then using ``metric='precomputed'`` here.

	Another way to reduce memory and computation time is to remove
	(near-)duplicate points and use ``sample_weight`` instead.

	:class:`~sklearn.cluster.OPTICS` provides a similar clustering with lower
	memory usage.

	References
	----------
	Ester, M., H. P. Kriegel, J. Sander, and X. Xu, `"A Density-Based
	Algorithm for Discovering Clusters in Large Spatial Databases with Noise"
	<https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf>`_.
	In: Proceedings of the 2nd International Conference on Knowledge Discovery
	and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

	Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017).
	:doi:`"DBSCAN revisited, revisited: why and how you should (still) use DBSCAN."
	<10.1145/3068335>`
	ACM Transactions on Database Systems (TODS), 42(3), 19.

	Examples
	--------
	>>> from sklearn.cluster import dbscan
	>>> X = [[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]
	>>> core_samples, labels = dbscan(X, eps=3, min_samples=2)
	>>> core_samples
	array([0, 1, 2, 3, 4])
	>>> labels
	array([ 0,  0,  0,  1,  1, -1])
	"""

	est = DBSCAN(
		eps=eps,
		min_samples=min_samples,
		metric=metric,
		metric_params=metric_params,
		algorithm=algorithm,
		leaf_size=leaf_size,
		p=p,
		n_jobs=n_jobs,
	)
	est.fit(X, sample_weight=sample_weight)
	return est.core_sample_indices_, est.labels_


class DBSCAN(ClusterMixin, BaseEstimator):
	"""Perform DBSCAN clustering from vector array or distance matrix.

	DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
	Finds core samples of high density and expands clusters from them.
	Good for data which contains clusters of similar density.

	This implementation has a worst case memory complexity of :math:`O({n}^2)`,
	which can occur when the `eps` param is large and `min_samples` is low,
	while the original DBSCAN only uses linear memory.
	For further details, see the Notes below.

	Read more in the :ref:`User Guide <dbscan>`.

	Parameters
	----------
	eps : float, default=0.5
		The maximum distance between two samples for one to be considered
		as in the neighborhood of the other. This is not a maximum bound
		on the distances of points within a cluster. This is the most
		important DBSCAN parameter to choose appropriately for your data set
		and distance function.

	min_samples : int, default=5
		The number of samples (or total weight) in a neighborhood for a point to
		be considered as a core point. This includes the point itself. If
		`min_samples` is set to a higher value, DBSCAN will find denser clusters,
		whereas if it is set to a lower value, the found clusters will be more
		sparse.

	metric : str, or callable, default='euclidean'
		The metric to use when calculating distance between instances in a
		feature array. If metric is a string or callable, it must be one of
		the options allowed by :func:`sklearn.metrics.pairwise_distances` for
		its metric parameter.
		If metric is "precomputed", X is assumed to be a distance matrix and
		must be square. X may be a :term:`sparse graph`, in which
		case only "nonzero" elements may be considered neighbors for DBSCAN.

		.. versionadded:: 0.17
		   metric *precomputed* to accept precomputed sparse matrix.

	metric_params : dict, default=None
		Additional keyword arguments for the metric function.

		.. versionadded:: 0.19

	algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
		The algorithm to be used by the NearestNeighbors module
		to compute pointwise distances and find nearest neighbors.
		See NearestNeighbors module documentation for details.

	leaf_size : int, default=30
		Leaf size passed to BallTree or cKDTree. This can affect the speed
		of the construction and query, as well as the memory required
		to store the tree. The optimal value depends
		on the nature of the problem.

	p : float, default=None
		The power of the Minkowski metric to be used to calculate distance
		between points. If None, then ``p=2`` (equivalent to the Euclidean
		distance).

	n_jobs : int, default=None
		The number of parallel jobs to run.
		``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
		``-1`` means using all processors. See :term:`Glossary <n_jobs>`
		for more details.

	Attributes
	----------
	core_sample_indices_ : ndarray of shape (n_core_samples,)
		Indices of core samples.

	components_ : ndarray of shape (n_core_samples, n_features)
		Copy of each core sample found by training.

	labels_ : ndarray of shape (n_samples)
		Cluster labels for each point in the dataset given to fit().
		Noisy samples are given the label -1.

	n_features_in_ : int
		Number of features seen during :term:`fit`.

		.. versionadded:: 0.24

	feature_names_in_ : ndarray of shape (`n_features_in_`,)
		Names of features seen during :term:`fit`. Defined only when `X`
		has feature names that are all strings.

		.. versionadded:: 1.0

	See Also
	--------
	OPTICS : A similar clustering at multiple values of eps. Our implementation
		is optimized for memory usage.

	Notes
	-----
	For an example, see
	:ref:`sphx_glr_auto_examples_cluster_plot_dbscan.py`.

	This implementation bulk-computes all neighborhood queries, which increases
	the memory complexity to O(n.d) where d is the average number of neighbors,
	while original DBSCAN had memory complexity O(n). It may attract a higher
	memory complexity when querying these nearest neighborhoods, depending
	on the ``algorithm``.

	One way to avoid the query complexity is to pre-compute sparse
	neighborhoods in chunks using
	:func:`NearestNeighbors.radius_neighbors_graph
	<sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
	``mode='distance'``, then using ``metric='precomputed'`` here.

	Another way to reduce memory and computation time is to remove
	(near-)duplicate points and use ``sample_weight`` instead.

	:class:`~sklearn.cluster.OPTICS` provides a similar clustering with lower memory
	usage.

	References
	----------
	Ester, M., H. P. Kriegel, J. Sander, and X. Xu, `"A Density-Based
	Algorithm for Discovering Clusters in Large Spatial Databases with Noise"
	<https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf>`_.
	In: Proceedings of the 2nd International Conference on Knowledge Discovery
	and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

	Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017).
	:doi:`"DBSCAN revisited, revisited: why and how you should (still) use DBSCAN."
	<10.1145/3068335>`
	ACM Transactions on Database Systems (TODS), 42(3), 19.

	Examples
	--------
	>>> from sklearn.cluster import DBSCAN
	>>> import numpy as np
	>>> X = np.array([[1, 2], [2, 2], [2, 3],
	...               [8, 7], [8, 8], [25, 80]])
	>>> clustering = DBSCAN(eps=3, min_samples=2).fit(X)
	>>> clustering.labels_
	array([ 0,  0,  0,  1,  1, -1])
	>>> clustering
	DBSCAN(eps=3, min_samples=2)
	"""

	_parameter_constraints: dict = {
		"eps": [Interval(Real, 0.0, None, closed="neither")],
		"min_samples": [Interval(Integral, 1, None, closed="left")],
		"metric": [
			StrOptions(set(_VALID_METRICS) | {"precomputed"}),
			callable,
		],
		"metric_params": [dict, None],
		"algorithm": [StrOptions({"auto", "ball_tree", "kd_tree", "brute"})],
		"leaf_size": [Interval(Integral, 1, None, closed="left")],
		"p": [Interval(Real, 0.0, None, closed="left"), None],
		"n_jobs": [Integral, None],
	}

	def __init__(
		self,
		eps=0.5,
		*,
		min_samples=5,
		metric="euclidean",
		metric_params=None,
		algorithm="auto",
		leaf_size=30,
		p=None,
		n_jobs=None,
	):
		self.eps = eps
		self.min_samples = min_samples
		self.metric = metric
		self.metric_params = metric_params
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.p = p
		self.n_jobs = n_jobs

	@_fit_context(
		# DBSCAN.metric is not validated yet
		prefer_skip_nested_validation=False
	)
	def fit(self, X, y=None, sample_weight=None):
		"""Perform DBSCAN clustering from features, or distance matrix.

		Parameters
		----------
		X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
			(n_samples, n_samples)
			Training instances to cluster, or distances between instances if
			``metric='precomputed'``. If a sparse matrix is provided, it will
			be converted into a sparse ``csr_matrix``.

		y : Ignored
			Not used, present here for API consistency by convention.

		sample_weight : array-like of shape (n_samples,), default=None
			Weight of each sample, such that a sample with a weight of at least
			``min_samples`` is by itself a core sample; a sample with a
			negative weight may inhibit its eps-neighbor from being core.
			Note that weights are absolute, and default to 1.

		Returns
		-------
		self : object
			Returns a fitted instance of self.
		"""
		# X = validate_data(self, X, accept_sparse="csr")

		if sample_weight is not None:
			sample_weight = _check_sample_weight(sample_weight, X)

		# Calculate neighborhood for all samples. This leaves the original
		# point in, which needs to be considered later (i.e. point i is in the
		# neighborhood of point i. While True, its useless information)
		if self.metric == "precomputed" and sparse.issparse(X):
			# set the diagonal to explicit values, as a point is its own
			# neighbor
			X = X.copy()  # copy to avoid in-place modification
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)
				X.setdiag(X.diagonal())

		neighbors_model = NearestNeighbors(
			radius=self.eps,
			algorithm=self.algorithm,
			leaf_size=self.leaf_size,
			metric=self.metric,
			metric_params=self.metric_params,
			p=self.p,
			n_jobs=self.n_jobs,
		)
		neighbors_model.fit(X)
		# This has worst case O(n^2) memory complexity
		neighborhoods = neighbors_model.radius_neighbors(X, return_distance=False)

		if sample_weight is None:
			n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])
		else:
			n_neighbors = np.array(
				[np.sum(sample_weight[neighbors]) for neighbors in neighborhoods]
			)

		# Initially, all samples are noise.
		labels = np.full(X.shape[0], -1, dtype=np.intp)

		# A list of all core samples found.
		core_samples = np.asarray(n_neighbors >= self.min_samples, dtype=np.uint8)
		dbscan_inner(core_samples, neighborhoods, labels)

		self.core_sample_indices_ = np.where(core_samples)[0]
		self.labels_ = labels

		if len(self.core_sample_indices_):
			# fix for scipy sparse indexing issue
			self.components_ = X[self.core_sample_indices_].copy()
		else:
			# no core samples
			self.components_ = np.empty((0, X.shape[1]))
		new_labels = []
		count = max(self.labels_)+1
	
		for value  in self.labels_:
			if value == -1:
				new_label = count + 1
				count +=1
			else:
				new_label = value
			new_labels.append(new_label)
		self.labels_ = new_labels
				
		return self

	def fit_predict(self, X, y=None, sample_weight=None):
		"""Compute clusters from a data or distance matrix and predict labels.

		Parameters
		----------
		X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
			(n_samples, n_samples)
			Training instances to cluster, or distances between instances if
			``metric='precomputed'``. If a sparse matrix is provided, it will
			be converted into a sparse ``csr_matrix``.

		y : Ignored
			Not used, present here for API consistency by convention.

		sample_weight : array-like of shape (n_samples,), default=None
			Weight of each sample, such that a sample with a weight of at least
			``min_samples`` is by itself a core sample; a sample with a
			negative weight may inhibit its eps-neighbor from being core.
			Note that weights are absolute, and default to 1.

		Returns
		-------
		labels : ndarray of shape (n_samples,)
			Cluster labels. Noisy samples are given the label -1.
		"""
		self.fit(X, sample_weight=sample_weight)
		return self.labels_

	def __sklearn_tags__(self):
		tags = super().__sklearn_tags__()
		tags.input_tags.pairwise = self.metric == "precomputed"
		tags.input_tags.sparse = True
		return tags

class DBScanPhuoc:
	def __init__(self,**kwargs):
		super(DBScanPhuoc, self).__init__(**kwargs)
		self.eps = 0.045# Adjust this value as needed (in normalized units)
		self.min_samples = 2 # Adjust this value as needed
		self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
	def normalize_centroids(self):
		self.normalized_centroids = np.copy(self.centroids)
		for poly in self.normalized_centroids:
			poly[ 0] = (poly[ 0])/self.w
			poly[ 1] = (poly[ 1]) /self.h
	def caculate_distance_matrix(self):
		n = len(self.normalized_centroids)
		self.distance_matrix = np.zeros((n, n), dtype=float)
		# Use NumPy's broadcasting to compute pairwise Euclidean distances efficiently
		diffs = self.normalized_centroids[:, np.newaxis, :] - self.normalized_centroids[np.newaxis, :, :]
		self.distance_matrix = np.sqrt(np.sum(diffs ** 2, axis=2))
		
	def caculate_orientation(self):
		self.vector_angles = []
		for poly in self.np_final_polys:
			x1,y1 = poly[0]
			x2,y2 = poly[1]
			self.vector_angles.append([x2-x1,y2-y1])

	def feetch_data(self,np_final_polys,image_size):
		self.w,self.h = image_size
		self.np_final_polys = np_final_polys
	
		self.final_polys = [Polygon(poly) for poly in self.np_final_polys]
		self.centroids = np.array([[poly.centroid.x, poly.centroid.y]  for poly in self.final_polys])
		
		self.normalize_centroids()
		self.caculate_orientation()
		self.caculate_distance_matrix()
		
		
		self.fit()
		if False:
			self.X_std = self.centroids	
			neighbors = len(self.X_std-1)
			nbrs = NearestNeighbors(n_neighbors=neighbors ).fit(np.copy(self.X_std))
			# Ma trận khoảng cách distances: (N, k)
			distances, indices = nbrs.kneighbors(self.X_std)
			# Lấy ra khoảng cách xa nhất từ phạm vi láng giềng của mỗi điểm và sắp xếp theo thứ tự giảm dần.
			distance_desc = sorted(distances[:, neighbors-1], reverse=True)
			# Vẽ biểu đồ khoảng cách xa nhất ở trên theo thứ tự giảm dần
			plt.figure(figsize=(12, 8))
			plt.plot(list(range(1,len(distance_desc )+1)), distance_desc)
			plt.axhline(y=0.12)
			plt.ylabel('distance')
			plt.xlabel('indice')
			plt.title('Sorting Maximum Distance in k Nearest Neighbor of kNN')
			import random 
			save_folder = "/work/21013187/phuoc/paddle_detect/checkpoints/det_db/tmp"
			import os
			os.makedirs(save_folder,exist_ok=True)
			plt.savefig(f"{save_folder}/knn_distance_plot_{random.uniform(0,10000)}.png", dpi=300, bbox_inches="tight")
			plt.close()
			plt.scatter(self.centroids[:,0], self.centroids[:,1], color='blue', marker='o')
			# Thêm tiêu đề và nhãn trục
			plt.title("2D Scatter Plot")
			plt.xlabel("X values")
			plt.ylabel("Y values")
			plt.savefig(f"{save_folder}/scatter_{random.uniform(0,10000)}.png", dpi=300, bbox_inches="tight")
			plt.close()
			exit()
	
	
	def visualize(self, image):
   

		# Loop through each polygon, centroid, and corresponding orientation vector
		for idx, (centroid, vector) in enumerate(zip(self.normalized_centroids, self.vector_angles)):
			cx, cy = int(centroid[0]*self.w), int(centroid[1]*self.h)  # Centroid coordinates
			vx, vy = vector[0], vector[1]  # Orientation vector components
			
			# Draw the centroid as a red filled circle
			cv2.circle(image, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)
			
			# Calculate the endpoint of the orientation arrow (scaled for visibility)
			scale = 100# Adjust this to make arrows longer or shorter
			try:
				end_x = int(cx + vx * scale / np.sqrt(vx**2 + vy**2))  # Normalize vector length
				end_y = int(cy + vy * scale / np.sqrt(vx**2 + vy**2))
			except:
				continue
			
			# Draw the orientation arrow from centroid to endpoint in green
			cv2.arrowedLine(image, (cx, cy), (end_x, end_y), color=(0, 255, 0), thickness=2, tipLength=0.2)

		scaled_centroids = self.centroids
		n = len(scaled_centroids)
		threshold = self.eps # Adjust this threshold as needed (in normalized units)
		for i in range(n):
			for j in range(i + 1, n):  # Avoid duplicate lines (i, j) and (j, i)
				if self.distance_matrix[i, j] < threshold:
					start_point = list(map(int,(scaled_centroids[i][0], scaled_centroids[i][1])))
					end_point = list(map(int,(scaled_centroids[j][0], scaled_centroids[j][1])))
					# Draw line in green
					cv2.line(image, start_point, end_point, color=(0, 255, 0), thickness=2)
					# Calculate midpoint for distance label
					mid_x, mid_y = (start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2
					# Add distance text
					distance_text = f"{self.distance_matrix[i, j]:.2f}"
					cv2.putText(image, distance_text, (mid_x, mid_y), 
								fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, 
								color=(0, 225, 0), thickness=2)
			# Return the modified frame
		return image
	def fit(self):
		dbscan = self.dbscan
		dbscan.fit(self.normalized_centroids)
		self.labels = dbscan.labels_
		
		self.core_samples_indices = dbscan.core_sample_indices_
		self.n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
		
  
	
		# return self.labels, self.core_samples_indices, self.n_clusters_
		