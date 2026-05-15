#! /bin/bash
NBITER=200
LOAD=False
NAME=pgd_demand
SNR=35
SEED=42
MODEL_LABEL=small.en
TEST_SPLIT=demand-100
CSV_NAME=demand-100
SKIP_PREP=False

python run_attack.py ./attack_configs/whisper/pgd_demand.yaml --root=./data --test_splits=$TEST_SPLIT --data_csv_name=$CSV_NAME --model_label=$MODEL_LABEL --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --snr=$SNR --skip_prep=$SKIP_PREP