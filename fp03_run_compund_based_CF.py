"""
Flavor-based item-item collaborative filtering: Figure 3 of
Palermo, Caprioli & Albora, "Food Pairing Unveiled" (arXiv:2406.15533).

Same prediction exercise as Figure 2, but the ingredient-ingredient similarity B is
computed on the ingredient-compound matrix F instead of the recipe-ingredient matrix R:

  1. drop recipes without ingredients and ingredients never used in any recipe;
  2. compute B = sim(F) for every method of Table 1 (F: ingredients x compounds,
     projection=0). B does not depend on the removed ingredients, so it is computed once;
  3. in each run, pick 10% of the recipes and remove one random ingredient from each;
  4. score the ingredients with Eq. 2,  S_ai = sum_j R_tilde_aj B_ji / sum_j |B_ji|;
  5. restrict the candidates to the CATEGORY of the removed ingredient (e.g. 'animal
     product' for 'yolk') by setting the other scores to 0, exclude the ingredients already in the recipe, and measure the
     rank of the removed one: Average Precision (= 1/rank) and Hit@K;
  6. repeat --n-iter times (100 in the paper) and plot mean +- one std across runs.

The random baseline draws uniform scores and goes through the same category restriction.

Inputs
  Data/A_rec_ingr.npy   recipes x ingredients (binary)
  Data/A_ingr_comp.npy  ingredients x compounds (binary), same ingredient order
  Data/ingr_info.tsv    Ahn et al.'s ingredient table ('# id', 'ingredient name',
                        'category'), needed for the category restriction

Usage
  python figure3.py
  python figure3.py --n-iter 10 --methods jaccard sapling random --seed 0
  python figure3.py --plot-only --output results_compounds
"""
import argparse
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from similarities import METHODS

warnings.filterwarnings("ignore")  # divisions by zero are handled by np.nan_to_num

LABELS = {name: label for name, (label, _) in METHODS.items()}
LABELS["random"] = "Random"


# ----------------------------------------------------------------------------- data

def load_data(rec_ingr_path, ingr_comp_path, ingr_info_path):
    """Load R, F and the ingredient categories, dropping unused ingredients as in the paper."""
    R = np.load(rec_ingr_path, mmap_mode="r")
    F = np.load(ingr_comp_path)
    if F.shape[0] != R.shape[1]:
        raise ValueError(f"A_ingr_comp has {F.shape[0]} ingredients (rows) but A_rec_ingr has "
                         f"{R.shape[1]} (columns): the two files must share the ingredient order.")

    R = np.asarray(R) > 0
    R = R[R.sum(axis=1) >= 1]                          # recipes with at least one ingredient
    used = np.flatnonzero(R.sum(axis=0) > 0)           # ingredients used in at least one recipe
    R = R[:, used].astype(np.float64)
    F = (F[used] > 0).astype(np.float64)

    categories = None
    if ingr_info_path is not None:
        info = pd.read_csv(ingr_info_path, sep="\t").rename(columns={"# id": "id"})
        cat = info.set_index("id")["category"].reindex(used)
        if cat.isna().any():
            raise ValueError(f"{ingr_info_path} has no category for ingredient ids "
                             f"{used[cat.isna().to_numpy()][:10].tolist()} (ids must match the "
                             "columns of A_rec_ingr).")
        categories = pd.factorize(cat.to_numpy())[0]

    return R, F, categories


def remove_ingredients(R, frac, rng):
    """Pick `frac` of the recipes and remove one random ingredient from each."""
    test_rec = rng.choice(R.shape[0], size=int(R.shape[0] * frac), replace=False)
    removed = np.array([rng.choice(np.flatnonzero(R[r])) for r in test_rec])
    R_test = R[test_rec].copy()
    R_test[np.arange(len(test_rec)), removed] = 0
    return R_test, removed


# ----------------------------------------------------------------------- evaluation

def evaluate_scores(S, R_test, removed, top_k, allowed=None, zero_outside=False, hr_ties="original"):
    """Average precision and hit@top_k for each modified recipe.

    `allowed` (n_test x n_ingr, bool) marks the candidates in the category of the removed
    ingredient. With zero_outside=True (default of the script, as in the code behind the
    paper) the score of the other candidates is set to 0; otherwise they are dropped from
    the ranking. The two coincide when the score of the removed ingredient is positive, and
    differ for metrics that can give zero or negative scores (Pearson, Sapling).
    Ingredients already in the recipe are always excluded.

    AP counts ties pessimistically (rank = number of candidates with score >= the removed
    one), which is what sklearn.metrics.average_precision_score returns; AP = 1 / rank.

    Hit@K: with hr_ties="original" the top K are taken with np.flip(np.argsort(scores)) on
    the candidate list, exactly as in the code behind the paper (ties broken by argsort's
    internal order); with hr_ties="pessimistic" the same rank as the AP is used. The two
    differ only when the removed ingredient ties with others around position K, which
    happens often here because every candidate outside the category scores 0.
    """
    S = S.astype(np.float64, copy=True)
    if allowed is not None:
        S[~allowed] = 0.0 if zero_outside else -np.inf
    S[R_test > 0] = -np.inf
    true_score = S[np.arange(len(removed)), removed]
    rank = np.sum(S >= true_score[:, None], axis=1)
    ap = 1.0 / rank
    if hr_ties == "pessimistic":
        return ap, rank <= top_k

    hit = np.empty(len(removed), dtype=bool)
    for j, r in enumerate(removed):
        keep = np.isfinite(S[j])                     # drop the ingredients already in the recipe
        pred = S[j][keep]
        pos = np.cumsum(keep)[r] - 1                 # index of the removed one after the deletion
        hit[j] = pos in np.flip(pred.argsort())[:top_k]
    return ap, hit


def run(args):
    rng = np.random.default_rng(args.seed)
    ingr_info = None if args.no_category else args.ingr_info
    if ingr_info is not None and not os.path.exists(ingr_info):
        raise FileNotFoundError(
            f"{ingr_info} not found. It is needed for the category restriction of Figure 3; "
            "pass its path with --ingr-info, or use --no-category to skip the restriction.")

    R, F, categories = load_data(args.rec_ingr, args.ingr_comp, ingr_info)
    n_rec, n_ingr = R.shape
    print(f"Dataset: {n_rec} recipes, {n_ingr} used ingredients, {F.shape[1]} compounds")
    if categories is None:
        print("WARNING: no category restriction, results will not match Figure 3.")

    # Flavor-based similarity: independent of the removed ingredients, computed once.
    B_norm = {}
    for name in args.methods:
        if name == "random":
            continue
        t0 = time.time()
        B = METHODS[name][1](F, 0)
        B_norm[name] = (B, np.sum(np.abs(B), axis=0))
        print(f"similarity {name:<24s} {time.time() - t0:.1f}s")

    rows = []
    for it in range(args.n_iter):
        # Same removed ingredients for all methods within a run (paired comparison)
        R_test, removed = remove_ingredients(R, args.frac, rng)
        allowed = None
        if categories is not None:
            allowed = categories[None, :] == categories[removed][:, None]

        for name in args.methods:
            if name == "random":
                S = rng.random(R_test.shape)
            else:
                B, norm = B_norm[name]
                S = np.nan_to_num(np.dot(R_test, B) / norm)          # Eq. 2
            ap, hit = evaluate_scores(S, R_test, removed, args.top_k, allowed,
                                      zero_outside=not args.exclude_outside_category,
                                      hr_ties=args.hr_ties)
            rows.append({"method": name, "iter": it, "mAP": ap.mean(), f"HR@{args.top_k}": hit.mean()})
        print(f"run {it + 1}/{args.n_iter}")

    per_run = pd.DataFrame(rows)
    #per_run.to_csv(f"{args.output}_per_run.csv", index=False)

    summary = per_run.groupby("method").agg(["mean", "std"]).drop(columns="iter")
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.sort_values("mAP_mean", ascending=False)
    summary.to_csv(f"output/{args.output}_summary.csv")
    print("\n" + summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved {args.output}_per_run.csv and {args.output}_summary.csv")


# -------------------------------------------------------------------------- figure

def plot(summary_path, figure_path):
    df = pd.read_csv(summary_path, index_col="method")
    hr_col = next(c[:-5] for c in df.columns if c.startswith("HR@") and c.endswith("_mean"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric, tag in zip(axes, ["mAP", hr_col], ["(a)", "(b)"]):
        d = df.sort_values(f"{metric}_mean")
        colors = ["red" if m == "random" else "tab:blue" for m in d.index]
        y = range(len(d))
        ax.errorbar(d[f"{metric}_mean"], y, xerr=d[f"{metric}_std"].fillna(0),
                    fmt="none", ecolor="gray", capsize=3)
        ax.scatter(d[f"{metric}_mean"], y, c=colors, zorder=3)
        ax.set_yticks(list(y))
        ax.set_yticklabels([LABELS.get(m, m) for m in d.index])
        ax.set_xlabel(metric)
        ax.set_title(tag, loc="left")
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    print(f"Saved {figure_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rec-ingr", default="Data/A_rec_ingr.npy", help="recipes x ingredients (.npy)")
    parser.add_argument("--ingr-comp", default="Data/A_ingr_comp.npy", help="ingredients x compounds (.npy)")
    parser.add_argument("--ingr-info", default="Data/ingr_info.tsv", help="ingredient categories (Ahn et al.)")
    parser.add_argument("--no-category", action="store_true",
                        help="do not restrict the candidates to the category of the removed ingredient")
    parser.add_argument("--exclude-outside-category", action="store_true",
                        help="drop the candidates outside the category from the ranking instead of "
                             "setting their score to 0 (the default, as in the code behind the paper)")
    parser.add_argument("--hr-ties", choices=["original", "pessimistic"], default="original",
                        help="tie handling in HR@K: argsort order as in the paper's code (default) "
                             "or pessimistic, consistent with the AP")
    parser.add_argument("--n-iter", type=int, default=100, help="number of runs (100 in the paper)")
    parser.add_argument("--frac", type=float, default=0.1, help="fraction of recipes with a removed ingredient")
    parser.add_argument("--top-k", type=int, default=20, help="K of the hit ratio (20 in the paper)")
    parser.add_argument("--methods", nargs="+", default=list(METHODS) + ["random"],
                        choices=list(METHODS) + ["random"], help="methods to evaluate (default: all)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--output", default="scores_compound_based", help="prefix of the output CSV files")
    parser.add_argument("--figure", default="figure3.pdf", help="output figure")
    parser.add_argument("--plot-only", action="store_true", help="only redraw the figure from the summary CSV")
    args = parser.parse_args()

    if not args.plot_only:
        run(args)
    plot(f"{args.output}_summary.csv", args.figure)


if __name__ == "__main__":
    main()
