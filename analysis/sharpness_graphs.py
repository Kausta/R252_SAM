import pandas as pd
from pyplot import plt
import numpy as np


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
    # custom order of run_description
    """run_description_order = [
        "128-sgd",
        "128-sam rho=0.05",
        "128-sam rho=0.1",
        "128-sam rho=0.2",
        "5000-sam rho=0.05",
        "5000-sam rho=0.1"
    ]
    df = df.sort_values(by="run_description", key=lambda x: run_description_order.index(x))"""
    df.plot.bar(x="run_description", y=["sharpness-5000-0.05", "sharpness-5000", "sharpness-5000-0.2", "sharpness-128-0.05", "sharpness-128", "sharpness-128-0.2", ""], ax=ax)
    # Legend title
    ax.legend(title="Sharpness Measure at Minima")
    # Set the labels and title
    ax.set_ylabel("Sharpness")
    ax.set_xlabel("Optimizer")
    ax.set_title("Final Minima Sharpness")
    # Save the figure as pdf
    fig.savefig("analysis/sharpness_graphs.pdf")


def bar_adversarial():
    # Read in the data
    df = pd.read_csv("analysis/sharpness_collection.csv")
    # Plot a barplot
    fig, ax = plt.subplots(figsize=(7, 4))
    # only if x and y are not nan and non None
    df = df.dropna(subset=["run_description", "sharpness-5000"])
    # drop duplicates based on run_description
    df = df.drop_duplicates(subset=["run_description"])
    # bar plot of both 'sharpness-5000' and 'sharpness-128'
    # custom order of run_description
    """run_description_order = [
        "128-sgd",
        "128-sam rho=0.05",
        "128-sam rho=0.1",
        "128-sam rho=0.2",
        "5000-sam rho=0.05",
        "5000-sam rho=0.1"
    ]
    df = df.sort_values(by="run_description", key=lambda x: run_description_order.index(x))"""
    df.plot.bar(x="run_description", y=["obj-5000-0.05", "obj-5000", "obj-5000-0.2", "obj-128-0.05", "obj-128", "obj-128-0.2"], ax=ax)
    # Legend title
    ax.legend(title="Sharpness Measure at Minima")
    # Set the labels and title
    ax.set_ylabel("Adversarial Training Loss")
    ax.set_xlabel("Optimizer")
    ax.set_title("Final Minima Adversarial Training Loss")
    # Save the figure as pdf
    fig.savefig("analysis/sharpness_graphs.pdf")


def sharpnesses_test_loss():
    df = pd.read_csv("analysis/sharpness_collection.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    # only if x and y are not nan and non None
    df = df.dropna(subset=["run_description", "sharpness-5000"])
    # drop duplicates based on run_description
    df = df.drop_duplicates(subset=["run_description"])
    ax.scatter(df["sharpness-128"], df["test_loss"], label="sharpness-128")
    ax.scatter(df["sharpness-5000"], df["test_loss"], label="sharpness-5000")
    ax.scatter(df["sharpness-128-0.05"], df["test_loss"], label="sharpness-128-0.05")
    ax.scatter(df["sharpness-5000-0.05"], df["test_loss"], label="sharpness-5000-0.05")
    ax.scatter(df["sharpness-128-0.2"], df["test_loss"], label="sharpness-128-0.2")
    ax.scatter(df["sharpness-5000-0.2"], df["test_loss"], label="sharpness-5000-0.2")
    ax.set_xlabel("Sharpness")
    ax.set_ylabel("Test Loss")
    ax.set_title("Minima Sharpness")
    ax.legend()
    fig.savefig("analysis/sharpness_graphs2.pdf")


def sharpnesses_test_loss_adjusted():
    df = pd.read_csv("analysis/sharpness_collection.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    # only if x and y are not nan and non None
    df = df.dropna(subset=["run_description", "sharpness-5000"])
    # drop duplicates based on run_description
    df = df.drop_duplicates(subset=["run_description"])
    ax.scatter(df["sharpness-128"], df["test_loss"] - df["obj-128"], label="sharpness-128")
    ax.scatter(df["sharpness-5000"], df["test_loss"] - df["obj-5000"], label="sharpness-5000")
    ax.scatter(df["sharpness-128-0.05"], df["test_loss"] - df["obj-128"], label="sharpness-128-0.05")
    ax.scatter(df["sharpness-5000-0.05"], df["test_loss"] - df["obj-5000"], label="sharpness-5000-0.05")
    ax.scatter(df["sharpness-128-0.2"], df["test_loss"] - df["obj-128"], label="sharpness-128-0.2")
    ax.scatter(df["sharpness-5000-0.2"], df["test_loss"] - df["obj-5000"], label="sharpness-5000-0.2")
    ax.set_xlabel("Sharpness")
    ax.set_ylabel("Test Loss - Adjusted")
    ax.set_title("Minima Sharpness")
    ax.legend()
    fig.savefig("analysis/sharpness_graphs2.pdf")


def adversarial_test_loss():
    df = pd.read_csv("analysis/sharpness_collection.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    # only if x and y are not nan and non None
    df = df.dropna(subset=["run_description", "sharpness-5000"])
    # drop duplicates based on run_description
    df = df.drop_duplicates(subset=["run_description"])

    # correlation between two columns
    corr = np.corrcoef(df["obj_orig-128"], df["test_loss"])[0, 1]
    corr_128 = np.corrcoef(df["obj-128"], df["test_loss"])[0, 1]
    corr_5000 = np.corrcoef(df["obj-5000"], df["test_loss"])[0, 1]
    corr_128_005 = np.corrcoef(df["obj-128-0.05"], df["test_loss"])[0, 1]
    corr_5000_005 = np.corrcoef(df["obj-5000-0.05"], df["test_loss"])[0, 1]
    corr_128_02 = np.corrcoef(df["obj-128-0.2"], df["test_loss"])[0, 1]
    corr_5000_02 = np.corrcoef(df["obj-5000-0.2"], df["test_loss"])[0, 1]

    ax.scatter(df["obj_orig-128"], df["test_loss"], label="train loss, corr: {:.2f}".format(corr))
    ax.scatter(df["obj-128"], df["test_loss"], label="sharpness-128, corr: {:.2f}".format(corr_128))
    ax.scatter(df["obj-5000"], df["test_loss"], label="sharpness-5000, corr: {:.2f}".format(corr_5000))
    ax.scatter(df["obj-128-0.05"], df["test_loss"], label="sharpness-128-0.05, corr: {:.2f}".format(corr_128_005))
    ax.scatter(df["obj-5000-0.05"], df["test_loss"], label="sharpness-5000-0.05, corr: {:.2f}".format(corr_5000_005))
    ax.scatter(df["obj-128-0.2"], df["test_loss"], label="sharpness-128-0.2, corr: {:.2f}".format(corr_128_02))
    ax.scatter(df["obj-5000-0.2"], df["test_loss"], label="sharpness-5000-0.2, corr: {:.2f}".format(corr_5000_02))

    ax.set_xlabel("Adversarial Training Loss")
    ax.set_ylabel("Test Loss")
    ax.set_title("Correlations")
    ax.legend()
    fig.savefig("analysis/sharpness_graphs3.pdf")

    # Is this interesting or no? Remember a lot of the algorithms didn't optimize for some of these measures at all. Thought the larger rho did well.


if __name__ == '__main__':
    bar_sharpness()