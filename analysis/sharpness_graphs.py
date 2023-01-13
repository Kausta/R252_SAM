import pandas as pd
from pyplot import plt


def bar_sharpness():
    # Read in the data
    df = pd.read_csv("analysis/sharpness_collection.csv")
    # Plot a barplot
    fig, ax = plt.subplots(figsize=(7, 4))
    # only if x and y are not nan and non None
    df = df.dropna(subset=["run_description", "sharpness-5000"])
    # drop duplicates based on run_description
    df = df.drop_duplicates(subset=["run_description"])
    # bar plot of both 'sharpness-5000' and 'sharpness-128'
    df.plot.bar(x="run_description", y=["sharpness-5000", "sharpness-128"], ax=ax)
    # Set the labels and title
    ax.set_ylabel("Sharpness")
    ax.set_xlabel("Optimizer")
    ax.set_title("Minima Sharpness")
    # Save the figure as pdf
    fig.savefig("analysis/sharpness_graphs.pdf")




if __name__ == '__main__':
    bar_sharpness()