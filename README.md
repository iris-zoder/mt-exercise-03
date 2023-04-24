# MT Exercise 3: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

## Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

## Steps

Clone this repository in the desired place:

    git clone https://github.com/siri-web/mt-exercise-03
    cd mt-exercise-03

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

### For task 1

**To reproduce my solution for the first task run the script download_holmes.sh:**

``
    ./scripts/download_holmes.sh
  ``

**Output**: The data will be saved under **data/holmes**, the model under **models/model_standard.pt**, and the generated
output under **samples/standard.txt**.

### For task 2

**To reproduce my solution for the second task run the script experiment_dropout.sh:**

``
    ./scripts/experiment_dropout.sh
  ``

**Output**: The log file with the different perplexities for each model will be saved under **logs/log_[DropRate].tsv**,
the corresponding model under **models/model_[DropRate].pt**, the tables and plots under **analysis**, and the samples
from the models with the highest/lowest perplexity under **samples/[highest/lowest]_ppl.txt** respectively.

## Changes to existing scripts

To make part two of the assignment easier I added pandas and seaborn to the scripts/install_packages.sh.
Although pandas was not strictly necessary I found it easier to work with - especially in combination
with seaborn for the plots.

Further, I changed the scripts/preprocess_raw.py file so that it cleans the data of certain specially characters
and story titles as those were generally not complete sentences and therefore not suited as training data for the
model in my opinion.

I changed the scripts/train.sh file to work for my machine and use my own data set.

I changed the scripts/generate.sh file to be called with two parameters - one for specifying the desired
model from which the text should be generated and the other for naming the sample:

  ``
    ./scripts/generate.sh MODEL_NAME=standard SAMPLE_NAME=standard.txt
  ``

Lastly I modified the tools/pytorch-examples/word_language_model/main.py script according to the exercise description
so that it now excepts the additional parameter --log-file to log the perplexities during training, validation and
testing.

## Added scripts

### Task 1

I added the bash script scripts/download_holmes.sh, which downloads the data I worked with, preprocesses it according
to scripts/download_data.sh (with some changes to account for the different text types), trains the model in the setting
most efficient for my machine and then generates some text using this model.

### Task 2

For the second part of the assignment I added the script scripts/experiment_dropout. When run, it traines models with 0,
0.2, 0.4, 0.6 and 0.8 dropout on my data, creates a log file for each under the logs directory, then calls the added python
script scripts/plot.py which saves the required tables and plots under the analysis directory. In a last step the script
generates text from the model with highest and lowest perplexity respectively and saves it under the samples directory.