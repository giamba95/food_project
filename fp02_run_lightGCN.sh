#!/bin/bash
set -e

# ---------- parameters ----------
DATASET_DIR="Data/train"
EMBED_DIR="Data/embeddings"
N_RUNS=100          # X goes from 0 to 99

# GCN hyperparameters (as requested)
EMBED_SIZE=64
LAYER_SIZE="[64,64,64,64,64,64]"   # 6 layers
LR=0.01
REGS="[1e-4]"
EPOCH=100
BATCH_SIZE=2048
# --------------------------------

mkdir -p "$EMBED_DIR"

# The data loader might also look for a test.txt: we create an empty
# one if it does not exist, so that it does not fail at startup.
touch "$DATASET_DIR/test.txt"

for X in $(seq 9 $((N_RUNS - 1))); do
    echo ""
    echo "=================================================="
    echo "=== Training $X / $((N_RUNS - 1)) su train_${X}.txt ==="
    echo "=================================================="

    TRAIN_FILE="$DATASET_DIR/train_${X}.txt"
    if [ ! -f "$TRAIN_FILE" ]; then
        echo "ATTENZIONE: $TRAIN_FILE non trovato, salto questa iterazione."
        continue
    fi

    # point train.txt to the current file (relative symlink)
    ln -sf "train_${X}.txt" "$DATASET_DIR/train.txt"
    
    python LightGCN.py \
        --dataset train \
        --data_path Data/ \
        --regs "$REGS" \
        --embed_size "$EMBED_SIZE" \
        --layer_size "$LAYER_SIZE" \
        --lr "$LR" \
        --batch_size "$BATCH_SIZE" \
        --epoch "$EPOCH"

    # rename the generated embeddings with index X
    mv "$EMBED_DIR/embeddings_users.csv" "$EMBED_DIR/embeddings_users_${X}.csv"
    mv "$EMBED_DIR/embeddings_items.csv" "$EMBED_DIR/embeddings_items_${X}.csv"

    echo "--- Salvati: ${EMBED_DIR}/embeddings_users_${X}.csv, ${EMBED_DIR}/embeddings_items_${X}.csv ---"
done

echo ""
echo "Tutti i training completati."
