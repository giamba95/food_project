# Food Pairing Unveiled – recommender systems on recipes, ingredients and flavor compounds

Code to reproduce the recommendation experiments of

> G. Palermo, C. Caprioli, G. Albora, *Food Pairing Unveiled: Exploring Recipe Creation Dynamics through Recommender Systems*, [arXiv:2406.15533](https://arxiv.org/abs/2406.15533) (2024)

The repository compares three ways of recommending ingredients to recipes on the dataset of Ahn et al. (*Flavor network and the principles of food pairing*, Sci. Rep. 1, 196, 2011):

- **Ingredient-based collaborative filtering.** Two ingredients are similar if they appear together in many recipes.
- **LightGCN.** A graph convolutional network trained on the recipe–ingredient network.
- **Compound-based collaborative filtering.** Two ingredients are similar if they share many flavor compounds, in line with the food pairing hypothesis.

All methods are evaluated with the same prediction exercise. One random ingredient is removed from 10% of the recipes, and each method has to retrieve it. Performance is measured with Mean Average Precision (mAP) and Hit Ratio at 20 (HR@20), averaged over 100 iterations with different random splits.

## Pipeline

Run the scripts in this order:

| Script | What it does | Output |
|---|---|---|
| `fp00_generate_train_and_test.py` | Generates the train/test splits of the 100 iterations. All the following scripts use these same splits, so every method is evaluated on the same removed ingredients. | train/test splits |
| `fp01_run_ingredient_based_CF.py` | Builds the collaborative filters based on the ingredient-based similarity (co-occurrences of ingredients in recipes), tests them on the 100 iterations and saves the average performance metrics. | `output/scores_ingredient_based_summary.csv` |
| `fp02_run_lightGCN.sh` | Runs LightGCN on the same splits and saves the average performance metrics. | `output/scores_lightGCN_summary.csv` |
| `fp03_run_compund_based_CF.py` | Builds the collaborative filters based on the compound-based similarity (flavor compounds shared by ingredients), tests them on the 100 iterations and saves the average performance metrics. | `output/scores_compound_based_summary.csv` |

```bash
python fp00_generate_train_and_test.py
python fp01_run_ingredient_based_CF.py
bash fp02_run_lightGCN.sh
python fp03_run_compund_based_CF.py
```

The results of `fp01` and `fp02` correspond to Figure 2 of the paper, and the results of `fp03` to Figure 3.

## Data

The scripts expect Ahn et al.'s dataset in the `Data/` folder:

- `A_rec_ingr.npy`: bi-adjacency matrix of the recipe–ingredient network (recipes on the rows, ingredients on the columns).
- `A_ingr_comp.npy`: bi-adjacency matrix of the ingredient–compound network (ingredients on the rows, flavor compounds on the columns), with the same ingredient order as `A_rec_ingr.npy`.
- `ingr_info.tsv`: ingredient names and categories. The compound-based recommender uses the categories to search for the removed ingredient only among ingredients of the same category.

## Methods

Both collaborative filters are item-item collaborative filters. For each metric of Table 1 of the paper, the scripts compute an ingredient–ingredient similarity matrix B:

- `fp01` computes B on the recipe–ingredient matrix.
- `fp03` computes B on the ingredient–compound matrix.

The score of ingredient *i* for recipe *α* is then

S<sub>αi</sub> = Σ<sub>j</sub> R<sub>αj</sub> B<sub>ji</sub> / Σ<sub>j</sub> |B<sub>ji</sub>|

where R is the recipe–ingredient matrix with the ingredients removed. The ingredients already in a recipe are excluded from its ranking.

The similarity metrics are Cooccurrences, Jaccard, Adamic/Adar, Resource Allocation, Cosine Similarity, Sorensen, Leicht-Holme-Newman, Hub Depressed, Hub Promoted, Taxonomy Network, Probabilistic Spreading, Pearson and Sapling Similarity. A random recommender is included as a baseline.

## Requirements

The experiments were run with the following library versions:

| Library | Version |
|---|---|
| scikit-learn | 0.19.1 |
| pandas | 2.1.1 |
| numpy | 1.26.0 |
| tensorflow | 1.11.0 |

TensorFlow 1.11 and scikit-learn 0.19.1 require Python ≤ 3.6, while pandas 2.1.1 and numpy 1.26.0 require Python ≥ 3.9. For this reason, LightGCN (`fp02`) has to run in a separate environment from the other scripts:

```bash
# Environment for fp00, fp01, fp03 (Python >= 3.9)
pip install numpy==1.26.0 pandas==2.1.1

# Environment for fp02 / LightGCN (Python 3.6)
pip install tensorflow==1.11.0 scikit-learn==0.19.1
```

## Citation

```bibtex
@article{palermo2024foodpairing,
  title   = {Food Pairing Unveiled: Exploring Recipe Creation Dynamics through Recommender Systems},
  author  = {Palermo, Giovanni and Caprioli, Claudio and Albora, Giambattista},
  journal = {arXiv preprint arXiv:2406.15533},
  year    = {2024}
}
```
