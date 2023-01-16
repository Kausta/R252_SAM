from pyplot import plt

import numpy as np

from model_map import models

if __name__ == "__main__":
    asym_all = []
    loss_all = []

    plt.figure(figsize=(5.5, 3))

    for i, model in enumerate(models.items()):
        name, desc = model
        with open("SAM-Checkpoints/%s/final-test-loss.txt" % name, "r") as f:
            loss = float(f.readline())

        with open("SAM-Checkpoints/%s/asymmetry-factor.txt" % name, "r") as f:
            asym = float(f.readline())

        if name == "fresh-darkness-163":
            continue

        asym_all.append(asym)
        loss_all.append(loss)

        plt.scatter(
            asym,
            loss,
            marker="o",
            s=30,
            label=desc["name"],
            color=desc["color"],
        )

    print(np.corrcoef(asym_all, loss_all))

    plt.plot(
        np.unique(asym_all),
        np.poly1d(np.polyfit(asym_all, loss_all, 1))(np.unique(asym_all)),
        color="salmon",
        linewidth=1,
    )
    plt.xlabel("Asymmetry Factor")
    plt.ylabel("Test Loss")
    plt.legend(ncol=1, bbox_to_anchor=(0.55, 0), loc="lower left")
    plt.savefig("asymmetry-plots/asymmetry-scatter.pdf")
