import pandas as pd
import seaborn as sns
import argparse
import os
from pathlib import Path
from tabulate import tabulate
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-directory", type=dir_path, help="Path to the directory with the log files to plot",
                        required=True)
    parser.add_argument("--output-directory", type=dir_path,
                        help="Directory where the tables and plots should be saved", required=True)

    args = parser.parse_args()

    return args


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        os.makedirs(path)
        return path


def main():
    args = parse_args()

    training_ppls = pd.DataFrame()
    validation_ppls = pd.DataFrame()
    test_ppls = pd.DataFrame()

    log_files = Path(args.log_directory).glob('*')

    for log_file in log_files:
        log_filename = str(log_file)
        setting = log_filename[log_filename.rindex("_") + 1:log_filename.rindex(".")]

        df = pd.read_csv(log_file, sep='\t', header=0)

        for total_df, column in zip([training_ppls, validation_ppls, test_ppls],
                                    ["training perplexity", "validation perplexity", "test perplexity"]):
            total_df[f"Dropout {setting}"] = df.loc[:, column]

    with open(os.path.join(args.output_directory, "tables"), 'w') as out_handle:
        for df, ppl in zip([training_ppls, validation_ppls, test_ppls],
                           ["training perplexity", "validation perplexity", "test perplexity"]):
            # make sure columns are sorted by dropout setting
            df.sort_index(axis=1, inplace=True)
            # shift indexes to match epochs, drop empty rows
            df.index += 1
            df = df.dropna()

            if ppl != "test perplexity":
                ax = sns.lineplot(data=df)
                ax.set(xlabel="Number of Epochs", ylabel="Perplexity", title=ppl)
                plt.legend(title="Dropout rate")
                plt.savefig(os.path.join(args.output_directory, ppl.replace(" ", "_") + "_plot.pdf"))
                plt.close()

            # do some renames for nicer tabulation
            df = df.set_index("Epoch " + df.index.astype(str))
            df.index.name = ppl
            out_handle.write(tabulate(df, headers="keys", stralign="right", numalign="right", floatfmt=".3f") + "\n\n")

        out_handle.write(f"The highest perplexity occurs with {test_ppls.dropna().idxmax(axis=1).loc[1]} and the lowest "
                         f"with {test_ppls.dropna().idxmin(axis=1).loc[1]}.")


if __name__ == "__main__":
    main()
