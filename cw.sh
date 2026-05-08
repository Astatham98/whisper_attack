#! /bin/bash
NBITER=2000
SEED=42
LOAD=False
NAME=cw
EPS=0.1
MAXDECR=8
CONF=0.0
DECRFACTOR=0.7
CST=4
LR=0.01
MODEL_LABEL=small
TEST_SPLIT=vctk-100
CSV_NAME=vctk-100
SKIP_PREP=True # Set to True after the first successful run to reuse cached files.

python run_attack.py ./attack_configs/whisper/cw_vctk.yml --root=./data --model_label=$MODEL_LABEL --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR --test_splits=$TEST_SPLIT --data_csv_name=$CSV_NAME --skip_prep=$SKIP_PREP
