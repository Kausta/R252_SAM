import matplotlib.colors as mcolours
from pyplot import plt

from model_map import models

if __name__ == "__main__":
    plt.figure(figsize=(5.5, 3))
    for i, model in enumerate(models.items()):
        name, desc = model
        X = []
        Y = []

        with open("SAM-Checkpoints/%s/asymmetry-loss-plot.txt" % name, "r") as f:
            for j, line in enumerate(f):
                if j == 0:
                    continue

                line = line.rstrip()
                x, y = line.split()

                X.append(float(x))
                Y.append(float(y))

        plt.scatter(0, Y[X.index(0)], marker="o", s=5, c=desc["color"])
        plt.plot(X, Y, color=desc["color"], label=desc["name"], linewidth=1)

    plt.xlabel(r"$\mu$")
    plt.legend(ncols=2)
    plt.ylabel(r"$L_\mathcal{S}(\bm{w}^* + \mu\bm{\hat{u}})$")
    plt.savefig("asymmetry-plots/asymmetry-all-loss.pdf")
