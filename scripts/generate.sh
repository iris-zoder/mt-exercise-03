#! /bin/bash

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done


scripts=$(dirname "$0")
base=$(realpath "$scripts"/..)

models=$base/models
data=$base/data
tools=$base/tools
samples=$base/samples

mkdir -p "$samples"

num_threads=8
device=""

(cd "$tools"/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data "$data"/holmes \
        --words 100 \
        --checkpoint "$models"/model_"$MODEL_NAME".pt \
        --outf "$samples"/"$SAMPLE_NAME".txt \
        --mps
)
