#!/bin/bash
# /data/yuanzi/codes/CIKM2025/NS4Rec_Framework/logs/WWW2026/LightGCN



# amazon
# ali
# yelp2018

# python main.py  --gnn lightgcn --lr 1e-03 --l2 1e-03 --dataset ali --pool mean --batch_size 2048 --gpu_id 0 --n_negs 1 --embedding0 False
# =================== 配置区 ===================
DATASETS=(yelp2018)
Embedding0=(False)
GNN=(igcn)
NEGS=1
CONTEXT_HOPS=(2)
POOL=(final mean sum concat)
B_MODE='sym'
S_MODE='cooccurrence'
S_NORM='sym'
GPU_ID=0
LOG_DIR="./logs/ItemGCN"
# =============================================

mkdir -p "$LOG_DIR"
TS=$(date +"%m%d_%H%M")


for i_data in "${DATASETS[@]}"; do
    for i_gnn in "${GNN[@]}"; do
        for i_embedding0 in "${Embedding0[@]}"; do
            for i_hops in "${CONTEXT_HOPS[@]}"; do

                # context_hops=0 时只有一层，因此 pooling 只跑一次
                if [ "$i_hops" -eq 0 ]; then
                    pool_list=(final)
                else
                    pool_list=("${POOL[@]}")
                fi

                for i_pool in "${pool_list[@]}"; do
                    LOG="${LOG_DIR}/${i_gnn}-${i_data}-hop${i_hops}-pool${i_pool}-negs${NEGS}-embedding0${i_embedding0}-${TS}.log"

                    echo "[$(date +%H:%M:%S)] START dataset=$i_data, gnn=$i_gnn, hops=$i_hops, pool=$i_pool, embedding0=$i_embedding0 → $LOG"

                    python main.py \
                        --gnn "$i_gnn" \
                        --dataset "$i_data" \
                        --pool "$i_pool" \
                        --n_negs "$NEGS" \
                        --context_hops "$i_hops" \
                        --b_mode "$B_MODE" \
                        --s_mode "$S_MODE" \
                        --s_norm "$S_NORM" \
                        --gpu_id "$GPU_ID" \
                        --embedding0 "$i_embedding0" \
                        > "$LOG" 2>&1

                    EXIT_CODE=$?
                    echo "[$(date +%H:%M:%S)] DONE dataset=$i_data, hops=$i_hops, pool=$i_pool, embedding0=$i_embedding0 (exit=$EXIT_CODE)"
                done
            done
        done
    done
done
