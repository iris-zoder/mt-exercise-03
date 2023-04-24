#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p "$data"

# download a different interesting data set!

mkdir -p "$data"/holmes

mkdir -p "$data"/holmes/raw

touch "$data"/holmes/raw/return.txt

wget -O "$data"/holmes/raw/return.txt https://sherlock-holm.es/stories/plain-text/retn.txt

# cut of lines with meta information so that they don't confuse the model
head -n 12950 "$data"/holmes/raw/return.txt | tail -n 12920 > "$data"/holmes/raw/return.cut.txt

# preprocess slightly
cat "$data"/holmes/raw/return.cut.txt | python "$base"/scripts/preprocess_raw.py > "$data"/holmes/raw/return.cleaned.txt

# tokenize, fix vocabulary upper bound

cat "$data"/holmes/raw/return.cleaned.txt | python "$base"/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    "$data"/holmes/raw/return.preprocessed.txt

# split into train, valid and test

head -n 600 "$data"/holmes/raw/return.preprocessed.txt | tail -n 600 > "$data"/holmes/valid.txt
head -n 1200 "$data"/holmes/raw/return.preprocessed.txt | tail -n 1200 > "$data"/holmes/test.txt
tail -n 5994 "$data"/holmes/raw/return.preprocessed.txt | head -n 5994 > "$data"/holmes/train.txt

/bin/bash "$scripts"/train.sh
/bin/bash "$scripts"/generate.sh MODEL_NAME=standard SAMPLE_NAME=standard
