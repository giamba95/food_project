import numpy as np
M = np.load('Data/M.npy')
n_users, n_items = M.shape

f = open('Data/edgelist_full_M.txt','w+')

item_idx = 0
item_map = {}
for u in range(n_users):
    row = ''
    row += '{}'.format(u)
    for i in range(n_items):
        if M[u,i] == 1:
            if i not in item_map.keys():
                item_map[i] = item_idx
                item_idx += 1
            row += ' {}'.format(item_map[i])
    f.write(row+'\n')
f.close()           

f = open('Data/columns_mapping.txt','w+')
for i in range(n_items):
    f.write('{} {}\n'.format(item_map[i],i))
f.close()

import os

# ---------- parameters ----------
INPUT_FILE = "Data/edgelist_full_M.txt"
OUTPUT_FILE = "Data/edgelist_full_M_clean.txt"

MIN_COLS = 2   # rows with fewer than MIN_COLS columns are removed
# --------------------------------

# 1) Read the original file
rows = []  # list of (old_row_index, [columns...])
with open(INPUT_FILE, "r") as f:
    for line in f:
        parts = line.split()
        if not parts:
            continue
        old_row_idx = int(parts[0])
        cols = [int(x) for x in parts[1:]]
        rows.append((old_row_idx, cols))

n_rows_before = len(rows)

# 2) Remove rows with 0 or 1 columns
kept_rows = [(old_idx, cols) for old_idx, cols in rows if len(cols) >= MIN_COLS]

n_rows_after = len(kept_rows)
n_removed = n_rows_before - n_rows_after

# 3) Determine the columns still present among the remaining rows,
#    and create a mapping old_column -> new_column (0..m-1),
#    preserving the original order (numerically ascending).
surviving_cols = sorted({c for _, cols in kept_rows for c in cols})
col_old_to_new = {old_c: new_c for new_c, old_c in enumerate(surviving_cols)}

# 4) Renumber the remaining rows from 0 to n_rows_after-1, in the order
#    in which they appear in the file (as if the removed rows had
#    never existed), and remap the columns accordingly.
row_old_to_new = {}
with open(OUTPUT_FILE, "w") as f:
    for new_row_idx, (old_row_idx, cols) in enumerate(kept_rows):
        row_old_to_new[old_row_idx] = new_row_idx
        new_cols = [col_old_to_new[c] for c in cols]  # original order preserved
        f.write(f"{new_row_idx} " + " ".join(str(c) for c in new_cols) + "\n")


print(f"Righe originali: {n_rows_before}")
print(f"Righe eliminate (0 o 1 colonna): {n_removed}")
print(f"Righe rimaste: {n_rows_after}")
print(f"Colonne originali distinte usate: {len(surviving_cols)}")
print(f"File pulito salvato in: {OUTPUT_FILE}")


import os
import random

# ---------- parameters ----------
INPUT_FILE  = "Data/edgelist_full_M_clean.txt"

FRACTION    = 0.10
SEED        = 42          # set to None to not fix the seed
N_ITER      = 100
# --------------------------------

if SEED is not None:
    random.seed(SEED)
for i in range(N_ITER):
    TRAIN_FILE  = "Data/train/train_{}.txt".format(i)
    TEST_FILE   = "Data/test/test_{}.txt".format(i)
    # 1) Read the file: each line -> (row_index, [columns...])
    rows = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            row_idx = int(parts[0])
            cols = [int(x) for x in parts[1:]]
            rows.append((row_idx, cols))

    n_rows = len(rows)

    # 2) Choose 10% of the rows from which to remove a column
    #    (only among the rows that have at least one column)
    eligible_indices = [i for i, (_, cols) in enumerate(rows) if len(cols) > 0]
    n_select = int(round(n_rows * FRACTION))
    n_select = min(n_select, len(eligible_indices))

    selected_indices = set(random.sample(eligible_indices, n_select))

    # 3) For each selected row, remove a randomly chosen column
    test_entries = []  # list of (row_idx, removed_col)

    for i in selected_indices:
        row_idx, cols = rows[i]
        removed_col = random.choice(cols)
        cols.remove(removed_col)
        rows[i] = (row_idx, cols)
        test_entries.append((row_idx, removed_col))

    # sort the test entries by row index (optional, but convenient)
    test_entries.sort(key=lambda x: x[0])

    # 4) Write train.txt (same format as the original file, without the removed columns)
    os.makedirs(os.path.dirname(TRAIN_FILE), exist_ok=True)
    with open(TRAIN_FILE, "w") as f:
        for row_idx, cols in rows:
            if cols:
                f.write(f"{row_idx} " + " ".join(str(c) for c in cols) + "\n")
            else:
                f.write(f"{row_idx}\n")

    # 5) Write test.txt with "row removed_column"
    os.makedirs(os.path.dirname(TEST_FILE), exist_ok=True)
    with open(TEST_FILE, "w") as f:
        for row_idx, removed_col in test_entries:
            f.write(f"{row_idx} {removed_col}\n")

    print(f"Righe totali: {n_rows}")
    print(f"Righe selezionate (10%): {n_select}")
    print(f"Train salvato in: {TRAIN_FILE}")
    print(f"Test salvato in: {TEST_FILE}")