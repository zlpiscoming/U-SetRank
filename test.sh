#!/bin/bash
ROOT_DIR=./
DATA_PATH=$ROOT_DIR/processed_data/lambda_mart_rank_data_istella_ndcg10_1_40_sh/
TEST_DIR=$ROOT_DIR/model/test_transformer_istella/
OUTPUT_DIR=/plsetrank/result

NAME="baseline with bias thread = 1.0"
echo ${NAME}
conda activate newplsetrank
sh ./scripts/train_transformer_istella.sh 0.002 True True True 1.0
conda activate insenv
sh ${ROOT_DIR}/scripts/evaluate_istella.sh ${TEST_DIR}/eval_test.ranklist | tee ${TEST_DIR}/result.log

