#! /bin/bash
NBITER=200
LOAD=False
NAME=pgd
SNR=35
SEED=42
MODEL_LABEL=small
TEST_SPLIT=vctk-8000
CSV_NAME=vctk-8000

python run_attack.py ./attack_configs/whisper/pgd_vctk.yaml --root=./data --test_splits=$TEST_SPLIT --data_csv_name=$CSV_NAME --model_label=$MODEL_LABEL --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --snr=$SNR