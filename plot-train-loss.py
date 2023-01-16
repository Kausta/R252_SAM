from pyplot import plt
import csv

from model_map import models

with open("stats/train-loss.csv") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        for key in models:
            if i == 0:
                models[key]["index"] = row.index(f"{key} - train/loss_epoch")
                models[key]["val_acc"] = []
            else:
                models[key]["val_acc"].append(float(row[models[key]["index"]]))

        if i == 200:
            break

X = [i for i in range(0, 200)]

for key in models:
    plt.plot(
        X,
        models[key]["val_acc"],
        color=models[key]["color"],
        label=models[key]["name"],
        linewidth=0.5,
    )

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.savefig("asymmetry-plots/train-loss.pdf")
