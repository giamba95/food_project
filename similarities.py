"""
Similarity metrics between the nodes of one layer of a bipartite network
(Table 1 of Palermo, Caprioli & Albora, "Food Pairing Unveiled", arXiv:2406.15533).

Every function takes
    M          : bi-adjacency matrix (rows = layer 0, columns = layer 1)
    projection : 0 -> similarity between ROWS of M
                 1 -> similarity between COLUMNS of M
and returns the square similarity matrix B.

In the paper M has recipes on the rows and ingredients on the columns, so the
ingredient-ingredient similarity is obtained with projection=1.

Formulas are identical to the original research code; the only change is that the
duplicated "projection == 0 / projection == 1" branches were merged by transposing M.
Reference for most definitions:
https://cran.r-project.org/web/packages/linkprediction/vignettes/proxfun.html
"""
import numpy as np


def _orient(M, projection):
    """Return X such that the similarity is always computed between the rows of X."""
    return M if projection == 0 else M.T


def cooccurrences(M, projection):
    X = _orient(M, projection)
    return np.dot(X, X.T)


def jaccard(M, projection):
    X = _orient(M, projection)
    k = np.sum(X, axis=1)
    coo = np.dot(X, X.T)
    return np.nan_to_num(coo / (np.subtract.outer(k, -k) - coo))


def adamic_adar(M, projection):
    X = _orient(M, projection)
    k = np.sum(X, axis=0)                 # degree of the nodes of the other layer
    k = np.nan_to_num(1 / np.log10(k))    # note: degree 1 -> 1/log(1) = inf (see README)
    return np.nan_to_num((X * k).dot(X.T))


def resource_allocation(M, projection):
    X = _orient(M, projection)
    k = np.sum(X, axis=0)
    return np.nan_to_num(np.nan_to_num(X / k).dot(X.T))


def cosine_similarity(M, projection):
    X = _orient(M, projection)
    k = np.sum(X, axis=1)
    coo = np.dot(X, X.T)
    return np.nan_to_num(coo / (np.multiply.outer(k, k) ** 0.5))


def sorensen(M, projection):
    X = _orient(M, projection)
    k = np.sum(X, axis=1)
    coo = np.dot(X, X.T)
    return np.nan_to_num(2 * coo / (np.subtract.outer(k, -k)))


def hub_depressed_index(M, projection):
    X = _orient(M, projection)
    k = np.sum(X, axis=1)
    coo = np.dot(X, X.T)
    return np.nan_to_num(coo / np.maximum.outer(k, k))


def hub_promoted_index(M, projection):
    X = _orient(M, projection)
    k = np.sum(X, axis=1)
    coo = np.dot(X, X.T)
    return np.nan_to_num(coo / np.minimum.outer(k, k))


def LHN(M, projection):
    """Leicht-Holme-Newman index."""
    X = _orient(M, projection)
    k = np.sum(X, axis=1)
    coo = np.dot(X, X.T)
    return np.nan_to_num(coo / (np.multiply.outer(k, k)))


def taxonomy_network(M, projection):
    """Zaccaria et al., 'How the taxonomy of products drives the economic development of countries'."""
    X = _orient(M, projection)
    k1 = np.sum(X, axis=1)
    k2 = np.sum(X, axis=0)
    return np.nan_to_num(np.nan_to_num(X / k2).dot(X.T) / np.maximum.outer(k1, k1))


def probabilistic_spreading(M, projection):
    """Zhou et al., 'Bipartite network projection and personal recommendation'."""
    X = _orient(M, projection)
    k1 = np.sum(X, axis=1)
    k2 = np.sum(X, axis=0)
    return np.nan_to_num(np.nan_to_num(X / k2).dot(X.T) / k1)


def pearson(M, projection):
    """Pearson correlation coefficient."""
    X = _orient(M, projection)
    k = np.sum(X, axis=1)
    coo = np.dot(X, X.T)
    N = X.shape[1]
    s = np.sum((X.T - k / N).T ** 2, axis=1) ** 0.5
    return np.nan_to_num((coo - np.multiply.outer(k, k) / N) / np.nan_to_num(np.multiply.outer(s, s)))


def sapling(M, projection):
    """Sapling similarity (Albora, Mori & Zaccaria, Knowledge-Based Systems 275, 2023)."""
    X = _orient(M, projection)
    N = X.shape[1]
    k = np.sum(X, axis=1)
    CO = np.dot(X, X.T)
    return np.nan_to_num(
        (1 - (CO * (1 - CO / k) + (k - CO.T).T * (1 - (k - CO.T).T / (N - k))).T / (k * (1 - k / N))).T
        * np.sign(((CO * N / k).T / k).T - 1)
    )


# Methods shown in Figure 2 (LightGCN excluded), with the labels used in the paper.
METHODS = {
    "cooccurrences": ("Cooccurrences", cooccurrences),
    "jaccard": ("Jaccard", jaccard),
    "adamic_adar": ("Adamic/Adar", adamic_adar),
    "resource_allocation": ("Resource Allocation", resource_allocation),
    "cosine_similarity": ("Cosine Similarity", cosine_similarity),
    "sorensen": ("Sorensen", sorensen),
    "LHN": ("LHN", LHN),
    "hub_depressed_index": ("Hub Depressed", hub_depressed_index),
    "hub_promoted_index": ("Hub Promoted", hub_promoted_index),
    "taxonomy_network": ("Taxonomy Network", taxonomy_network),
    "probabilistic_spreading": ("Probabilistic Spreading", probabilistic_spreading),
    "pearson": ("Pearson", pearson),
    "sapling": ("Sapling", sapling),
}
