from pyplot import plt

from model_map import models

if __name__ == "__main__":
    for i, model in enumerate(models.items()):
        name, desc = model
        X = []
        Y = []

        with open("SAM-Checkpoints/%s/asymmetry-loss-plot.txt" % name, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue

                line = line.rstrip()
                x, y = line.split()

                X.append(float(x))
                Y.append(float(y))

        print(max(X))
        plt.figure(figsize=(1.375, 1.375))
        plt.plot(X, Y, label="Training Loss", color="dodgerblue")
        plt.scatter(0, Y[X.index(0)], marker="o", s=20, c="orange", label=r"$\bm{w}^*$")
        # plt.xlabel(r"$\mu$")
        # plt.legend()
        # plt.ylabel(r"$L_\mathcal{S}(\bm{w}^* + \mu\bm{\hat{u}})$")
        plt.xticks([])
        plt.yticks([])
        plt.savefig("asymmetry-plots/%s-asymmetry-loss-plot.pdf" % name)
        plt.clf()
