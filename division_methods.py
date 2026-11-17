from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import numpy as np

# def random_sampling(features, target, test_size):
#     """ Random sampling using scikit-learn's train_test_split. """
#     return train_test_split(features, target, test_size=test_size, random_state=42)

# def kennard_stone_sampling(features, target, test_size):
#     """ Kennard-Stone sampling using the kennard-stone package. """
#     return ks_train_test_split(features, target, test_size=test_size)

# def euclidean_sampling(features, target, test_size):
#     """ Custom Euclidean-based sampling to maximize distances between selected samples. """
#     n_samples = len(features)
#     train_size = int((1 - test_size) * n_samples)
#     distances = cdist(features, features, 'euclidean')
#     np.fill_diagonal(distances, np.inf)
    
#     train_indices = [np.random.randint(n_samples)]
#     while len(train_indices) < train_size:
#         remaining_indices = list(set(range(n_samples)) - set(train_indices))
#         dist_to_selected = distances[train_indices][:, remaining_indices]
#         next_index = remaining_indices[np.argmax(np.min(dist_to_selected, axis=0))]
#         train_indices.append(next_index)
    
#     test_indices = list(set(range(n_samples)) - set(train_indices))
#     return features.iloc[train_indices], features.iloc[test_indices], target.iloc[train_indices], target.iloc[test_indices]


# /////////////////////////////////////////////////////////////////
# def kennard_stone_sampling(X, y, test_size):
#     # Implement or import your Kennard-Stone algorithm here
#     test_indices = np.random.choice(X.shape[0], size=int(X.shape[0] * test_size), replace=False)
#     train_indices = list(set(range(X.shape[0])) - set(test_indices))
#     return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# def euclidean_samplin(X, y, test_size):
#     # Implement or import your Euclidean sampling algorithm here
#     test_indices = np.random.choice(X.shape[0], size=int(X.shape[0] * test_size), replace=False)
#     train_indices = list(set(range(X.shape[0])) - set(test_indices))
#     return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
# /////////////////////////////////////////////////////////////////


# from sklearn.cluster import KMeans
# from sklearn.model_selection import train_test_split



def euclidean_sampling(X, y, test_size):
# Create a KMeans model with the desired number of clusters (e.g., 2)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)

    # Assign cluster labels to each data point
    cluster_labels = kmeans.labels_

    # Split the data based on cluster labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=cluster_labels, test_size=test_size, random_state=42)
    
    return  X_train, X_test, y_train, y_test

# /////////////////////////////////////////////////////////////////


def kennard_stone_sampling(X, y, test_size):
    """
    Kennard-Stone algorithm for selecting a representative subset of samples.

    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.
    y : array-like of shape (n_samples,)
        Target labels for each sample in X.
    test_size : float, optional (default=0.2)
        Proportion of samples to use in the test set. Must be between 0 and 1.

    Returns:
    --------
    X_train : ndarray of shape (n_train_samples, n_features)
        Training set samples.
    X_test : ndarray of shape (n_test_samples, n_features)
        Test set samples.
    y_train : array-like of shape (n_train_samples,)
        Training set labels.
    y_test : array-like of shape (n_test_samples,)
        Test set labels.
    """
    # Compute the pairwise distance matrix
    distances = pairwise_distances(X, X)

    # Initialize the lists
    selected_indices = [np.argmax(np.mean(distances, axis=1))]  # Start with the most central sample
    remaining_indices = [i for i in range(X.shape[0]) if i != selected_indices[0]]

    # Select samples until reaching the desired number of training samples
    while len(selected_indices) < int(X.shape[0] * (1 - test_size)):
        # Find the sample with the maximum minimum distance to selected samples
        min_distances = distances[remaining_indices][:, selected_indices].min(axis=1)
        max_idx = remaining_indices[np.argmax(min_distances)]

        # Add to the selected indices and remove from the remaining set
        selected_indices.append(max_idx)
        remaining_indices.remove(max_idx)

    # Split the data into training and test sets
    X_train = X[selected_indices]
    X_test = X[remaining_indices]
    y_train = y[selected_indices]
    y_test = y[remaining_indices]

    return X_train, X_test, y_train, y_test
