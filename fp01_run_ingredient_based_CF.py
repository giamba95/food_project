"""
Recipe-based item-item collaborative filtering: prediction exercise of Figure 2 in
Palermo, Caprioli & Albora, "Food Pairing Unveiled" (arXiv:2406.15533).

For each of the 100 folds:
  1. the recipe-ingredient network is built from Data/train/train_<fold>.txt, i.e. the
     matrix R_tilde in which one ingredient per evaluated recipe has been removed;
  2. the ingredient-ingredient similarity B is computed on R_tilde with each metric of
     Table 1;
  3. the recommendation scores are computed with Eq. 2,
         S_ai = sum_j R_tilde_aj B_ji / sum_j |B_ji|
     for the recipes appearing in Data/test/test_<fold>.txt;
  4. the ingredients already in the recipe (i.e. in the train file) are excluded from the
     ranking, and the position of the held-out ingredient(s) of the test file gives the
     Average Precision and the hit@K of that recipe.
mAP and HR@K are the averages over the test recipes of a fold; mean and standard
deviation are then taken over the 100 folds.

Usage:
    python evaluate.py --data-dir Data
    python evaluate.py --data-dir Data --folds 0 9 --methods jaccard sapling random
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd

from data import build_matrix, dataset_shape, fold_paths, read_interactions
from similarities import METHODS

warnings.filterwarnings("ignore")  # divisions by zero are handled by np.nan_to_num


def recommendation_scores(R_test, B):
    """Eq. 2: S = R_test B / column-wise sum of |B|."""
    return np.nan_to_num(np.dot(R_test, B) / np.sum(np.abs(B), axis=0))


def evaluate_scores(S, known, held_out, top_k):
    """Average precision and hit@top_k of each test recipe.

    `known` is the binary matrix of the ingredients already in the recipes (train file):
    they are excluded from the ranking. `held_out` is the list of the ingredients to
    retrieve for each recipe.

    Ties are counted pessimistically: the rank of a held-out ingredient is the number of
    candidates with a score greater than or equal to its own. With a single held-out
    ingredient AP = 1 / rank, which is what sklearn's average_precision_score returns.
    """
    S = S.astype(np.float64, copy=True)
    S[known > 0] = -np.inf

    n_held = np.array([len(h) for h in held_out])
    if np.all(n_held == 1):                       # fast path: one ingredient per recipe
        targets = np.array([h[0] for h in held_out])
        rank = np.sum(S >= S[np.arange(len(targets)), targets][:, None], axis=1)
        return 1.0 / rank, rank <= top_k

    ap = np.empty(len(held_out))
    hit = np.empty(len(held_out), dtype=bool)
    for j, items in enumerate(held_out):
        ap[j] = _average_precision(S[j], items)
        hit[j] = np.min(np.sum(S[j] >= S[j][items][:, None], axis=1)) <= top_k
    return ap, hit


def _average_precision(scores, targets):
    """Average precision of one recipe with several held-out ingredients.

    Same definition as sklearn.metrics.average_precision_score: precision and recall are
    evaluated at each distinct score threshold and AP = sum_k (R_k - R_k-1) * P_k.
    """
    relevant = np.zeros(len(scores), dtype=bool)
    relevant[targets] = True
    order = np.argsort(-scores, kind="stable")
    scores, relevant = scores[order], relevant[order]

    tp = np.cumsum(relevant)
    seen = np.arange(1, len(scores) + 1)
    last = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]  # end of each tie group
    precision = tp[last] / seen[last]
    recall = tp[last] / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def run_fold(data_dir, fold, shape, methods, top_k, rng):
    train_path, test_path = fold_paths(data_dir, fold)
    train = read_interactions(train_path)
    test = read_interactions(test_path)

    R = build_matrix(train, shape)
    test_recipes = np.array(sorted(test))
    held_out = [test[r] for r in test_recipes]
    R_test = R[test_recipes]

    out = {}
    for name in methods:
        if name == "random":
            S = rng.random(R_test.shape)
        else:
            B = METHODS[name][1](R, 1)            # ingredient-ingredient similarity
            S = recommendation_scores(R_test, B)
        ap, hit = evaluate_scores(S, R_test, held_out, top_k)
        out[name] = (ap.mean(), hit.mean())
    return out, len(test_recipes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="Data", help="folder containing train/ and test/")
    parser.add_argument("--folds", type=int, nargs=2, default=(0, 99), metavar=("FIRST", "LAST"),
                        help="range of folds to evaluate, inclusive (default: 0 99)")
    parser.add_argument("--top-k", type=int, default=20, help="K of the hit ratio (20 in the paper)")
    parser.add_argument("--methods", nargs="+", default=list(METHODS) + ["random"],
                        choices=list(METHODS) + ["random"], help="methods to evaluate (default: all)")
    parser.add_argument("--seed", type=int, default=None, help="seed of the random baseline")
    parser.add_argument("--output", default="scores_ingredient_based", help="prefix of the output CSV files")
    args = parser.parse_args()

    folds = range(args.folds[0], args.folds[1] + 1)
    rng = np.random.default_rng(args.seed)

    shape = dataset_shape(args.data_dir, folds)
    print(f"Network size: {shape[0]} recipes x {shape[1]} ingredients")

    rows = []
    for fold in folds:
        t0 = time.time()
        results, n_test = run_fold(args.data_dir, fold, shape, args.methods, args.top_k, rng)
        for name, (mean_ap, hit_ratio) in results.items():
            rows.append({"method": name, "fold": fold, "n_test_recipes": n_test,
                         "mAP": mean_ap, f"HR@{args.top_k}": hit_ratio})
        print(f"fold {fold} ({n_test} test recipes) done in {time.time() - t0:.1f}s")

    per_fold = pd.DataFrame(rows)
    #per_fold.to_csv(f"output/{args.output}_per_fold.csv", index=False)

    summary = per_fold.groupby("method")[["mAP", f"HR@{args.top_k}"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.sort_values("mAP_mean", ascending=False)
    summary.to_csv(f"output/{args.output}_summary.csv")

    print("\n" + summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved {args.output}_per_fold.csv and {args.output}_summary.csv")


if __name__ == "__main__":
    main()
