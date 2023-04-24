#!/bin/bash

scripts=$(dirname "$0")
base=$(realpath "$scripts"/..)

models=$base/models
logs=$base/logs
data=$base/data
tools=$base/tools

mkdir -p "$models"
mkdir -p "$logs"

num_threads=8
device=""

for drop in $(seq 0 0.2 0.8)
do
  echo "Training model with $drop dropout rate"
  SECONDS=0
  # I set the log-interval to 128 as there are 128 batches for my training set and I wanted to have the ppl at the end
  # of a training epoch
  (cd "$tools"/pytorch-examples/word_language_model &&
  CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data "$data"/holmes \
  --epochs 40 \
  --log-interval 149 \
  --emsize 200 --nhid 200 --tied \
  --dropout "$drop" \
  --save "$models"/model_"$drop".pt \
  --log-file "$logs"/log_"$drop".tsv \
  --mps
  )
    echo "time taken:"
    echo "$SECONDS seconds"
done

python3 "$scripts"/plot.py --log-directory "$logs" --output-directory "$base"/analysis

highest_lowest=($(tail -n 1 "$base"/analysis/tables | ggrep -Poh '\d+\.?\d+'))
/bin/bash "$scripts"/generate.sh MODEL_NAME="${highest_lowest[0]}" SAMPLE_NAME=highest_ppl
/bin/bash "$scripts"/generate.sh MODEL_NAME="${highest_lowest[1]}" SAMPLE_NAME=lowest_ppl