set -e
set -x

TAG=click_lambda_mart_output_istella_1000_20_ndcg10
OUTPUT_DIR=./processed_data/${TAG}/
mkdir -p $OUTPUT_DIR

ROOT_PATH=/home/leping_zhang/data/istella_letor/split
TRAIN_FILE=$ROOT_PATH/newtrain.txt
TEST_FILE=$ROOT_PATH/test.txt.norm
java -jar ../ranklib/RankLib-2.12.jar -train $TRAIN_FILE -ranker 0 -test $TEST_FILE -tree $1 -metric2T NDCG@$2 -leaf 20

