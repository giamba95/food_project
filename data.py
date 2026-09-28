"""
Reading of the train/test files (LightGCN-style format).

Each line is
    <recipe_id> <ingredient_id> <ingredient_id> ...
The train file holds the recipes with the held-out ingredient(s) already removed
(the matrix R_tilde of the paper), the test file holds the held-out ingredient(s)
of the evaluated recipes.
"""
import os

import numpy as np


def read_interactions(path):
    """Read one file and return {recipe_id: array of ingredient ids}.

    Lines with only the recipe id (no ingredients) are skipped.
    """
    interactions = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            recipe = int(parts[0])
            items = np.unique(np.array(parts[1:], dtype=np.int64))
            interactions[recipe] = items
    return interactions


def fold_paths(data_dir, fold):
    return (os.path.join(data_dir, "train", f"train_{fold}.txt"),
            os.path.join(data_dir, "test", f"test_{fold}.txt"))


def dataset_shape(data_dir, folds):
    """Largest recipe and ingredient id over all folds, so that every fold uses
    matrices of the same size and the results stay comparable."""
    max_recipe = max_ingredient = -1
    for fold in folds:
        for path in fold_paths(data_dir, fold):
            for recipe, items in read_interactions(path).items():
                max_recipe = max(max_recipe, recipe)
                max_ingredient = max(max_ingredient, int(items.max()))
    return max_recipe + 1, max_ingredient + 1


def build_matrix(interactions, shape):
    """Bi-adjacency matrix: recipes on the rows, ingredients on the columns."""
    R = np.zeros(shape, dtype=np.float64)
    for recipe, items in interactions.items():
        R[recipe, items] = 1.0
    return R
