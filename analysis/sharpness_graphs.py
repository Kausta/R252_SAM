import pandas as pd
from pyplot import plt
import numpy as np

# A lot of this code was generated with copilot.


def bar_sharpness():
    sharpness_metric = "shrp (128, 0.1) "
    # Read in the data
    df = pd.read_csv("analysis/sharpness_collection.csv")
    # Plot a barplot
    fig, ax = plt.subplots(figsize=(7, 4))
    # only if x and y are not nan and non None
    df = df.dropna(subset=["run_description", 'avg loss (5000, 1)'])
    # sort df by run_description
    df = df.sort_values(by="run_description")
    # drop duplicates based on run_description
    df = df.drop_duplicates(subset=["run_description"])
    # bar plot of both 'sharpness-5000' and 'sharpness-128'
    df.plot.bar(x="run_description", y=sharpness_metric, ax=ax)
    # no legend
    ax.legend().set_visible(False)
    # Set the labels and title
    ax.set_ylabel("Sharpness")
    ax.set_xlabel("Optimizer")
    # title
    ax.set_title("Batch Size 128")
    # Save the figure as pdf
    fig.savefig("analysis/sharpness_graphs_0.pdf")


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
    df = df.dropna(subset=["run_description", "avg loss (128, 1)"])
    # drop duplicates based on run_description
    df = df.drop_duplicates(subset=["run_description"])


    # correlation between two columns
    corr_128 = np.corrcoef(df["adv loss (128, 0.1)"], df["test loss"])[0, 1]
    corr_5000 = np.corrcoef(df["adv loss (5000, 0.1)"], df["test loss"])[0, 1]
    corr_128_005 = np.corrcoef(df["adv loss (128, 0.05)"], df["test loss"])[0, 1]
    corr_5000_005 = np.corrcoef(df["adv loss (5000, 0.05)"], df["test loss"])[0, 1]
    corr_128_02 = np.corrcoef(df["adv loss (128, 0.2)"], df["test loss"])[0, 1]
    corr_5000_02 = np.corrcoef(df["adv loss (5000, 0.2)"], df["test loss"])[0, 1]

    s = 15

    ax.scatter(df["adv loss (128, 0.1)"], df["test loss"], label="m: 128, $\\rho$: 0.1, corr: {:.2f}".format(corr_128), s=s)
    m, b = np.polyfit(df["adv loss (128, 0.1)"], df["test loss"], 1)
    ax.plot(df["adv loss (128, 0.1)"], m*df["adv loss (128, 0.1)"] + b, color="C0", linewidth=1)
    ax.scatter(df["adv loss (5000, 0.1)"], df["test loss"], label="m: 5000, $\\rho$: 0.1, corr: {:.2f}".format(corr_5000), s=s)
    m, b = np.polyfit(df["adv loss (5000, 0.1)"], df["test loss"], 1)
    ax.plot(df["adv loss (5000, 0.1)"], m*df["adv loss (5000, 0.1)"] + b, color="C1", linewidth=1)
    ax.scatter(df["adv loss (128, 0.05)"], df["test loss"], label="m: 128, $\\rho$: 0.05, corr: {:.2f}".format(corr_128_005), s=s)
    m, b = np.polyfit(df["adv loss (128, 0.05)"], df["test loss"], 1)
    ax.plot(df["adv loss (128, 0.05)"], m*df["adv loss (128, 0.05)"] + b, color="C2", linewidth=1)
    ax.scatter(df["adv loss (5000, 0.05)"], df["test loss"], label="m: 5000, $\\rho$: 0.05, corr: {:.2f}".format(corr_5000_005), s=s)
    m, b = np.polyfit(df["adv loss (5000, 0.05)"], df["test loss"], 1)
    ax.plot(df["adv loss (5000, 0.05)"], m*df["adv loss (5000, 0.05)"] + b, color="C3", linewidth=1)
    ax.scatter(df["adv loss (128, 0.2)"], df["test loss"], label="m: 128, $\\rho$: 0.2, corr: {:.2f}".format(corr_128_02), s=s)
    m, b = np.polyfit(df["adv loss (128, 0.2)"], df["test loss"], 1)
    ax.plot(df["adv loss (128, 0.2)"], m*df["adv loss (128, 0.2)"] + b, color="C4", linewidth=1)
    ax.scatter(df["adv loss (5000, 0.2)"], df["test loss"], label="m: 5000, $\\rho$: 0.2, corr: {:.2f}".format(corr_5000_02), s=s)
    m, b = np.polyfit(df["adv loss (5000, 0.2)"], df["test loss"], 1)
    ax.plot(df["adv loss (5000, 0.2)"], m*df["adv loss (5000, 0.2)"] + b, color="C5", linewidth=1)

    ax.set_xlabel("Adversarial Perturbation Training Loss")
    ax.set_ylabel("Test Loss")
    ax.legend(title="Measure for Adversarial Perturbation Loss")
    fig.savefig("analysis/adv_loss.pdf")


def average_test_loss():
    df = pd.read_csv("analysis/sharpness_collection.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    # only if x and y are not nan and non None
    df = df.dropna(subset=["run_description", "avg loss (128, 1)"])
    # drop duplicates based on run_description
    df = df.drop_duplicates(subset=["run_description"])


    # correlation between two columns
    corr = np.corrcoef(df["obj_orig-128"], df["test loss"])[0, 1]
    corr_avg_5000_05 = np.corrcoef(df["avg loss (5000, 0.5)"], df["test loss"])[0, 1]
    corr_avg_128_1 = np.corrcoef(df["avg loss (128, 05)"], df["test loss"])[0, 1]
    corr_avg_5000_1 = np.corrcoef(df["avg loss (5000, 1)"], df["test loss"])[0, 1]
    corr_avg_128_2 = np.corrcoef(df["avg loss (128, 1)"], df["test loss"])[0, 1]

    ax.scatter(df["obj_orig-128"], df["test loss"], label="regular train loss, corr: {:.2f}".format(corr), s=30)
    m, b = np.polyfit(df["obj_orig-128"], df["test loss"], 1)
    ax.plot(df["obj_orig-128"], m*df["obj_orig-128"] + b, color="C0", linewidth=1)
    ax.scatter(df["avg loss (5000, 0.5)"], df["test loss"], label="m: 5000, rho: 0.5, corr: {:.2f}".format(corr_avg_5000_05), s=30)
    m, b = np.polyfit(df["avg loss (5000, 0.5)"], df["test loss"], 1)
    ax.plot(df["avg loss (5000, 0.5)"], m*df["avg loss (5000, 0.5)"] + b, color="C1", linewidth=1)
    ax.scatter(df["avg loss (128, 05)"], df["test loss"], label="m: 128, rho: 0.5, corr: {:.2f}".format(corr_avg_128_1), s=30)
    m, b = np.polyfit(df["avg loss (128, 05)"], df["test loss"], 1)
    ax.plot(df["avg loss (128, 05)"], m*df["avg loss (128, 05)"] + b, color="C2", linewidth=1)
    ax.scatter(df["avg loss (5000, 1)"], df["test loss"], label="m: 5000, rho: 1, corr: {:.2f}".format(corr_avg_5000_1), s=30)
    m, b = np.polyfit(df["avg loss (5000, 1)"], df["test loss"], 1)
    ax.plot(df["avg loss (5000, 1)"], m*df["avg loss (5000, 1)"] + b, color="C3", linewidth=1)
    ax.scatter(df["avg loss (128, 1)"], df["test loss"], label="m: 128, rho: 1, corr: {:.2f}".format(corr_avg_128_2), s=30)
    m, b = np.polyfit(df["avg loss (128, 1)"], df["test loss"], 1)
    ax.plot(df["avg loss (128, 1)"], m*df["avg loss (128, 1)"] + b, color="C4", linewidth=1)

    ax.set_xlabel("Average Perturbation Training Loss")
    ax.set_ylabel("Test Loss")
    ax.legend(title="Average Perturbation Measure")
    fig.savefig("analysis/sharpness_graphs4.pdf")


if __name__ == '__main__':
    adversarial_test_loss()