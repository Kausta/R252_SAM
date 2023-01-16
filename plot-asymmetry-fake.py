from pyplot import plt
import numpy as np


def f1(x):
    if x <= 1 and x >= -1:
        return 0

    if x > 1:
        return 0.1 * (x - 1) ** 2

    if x < -1:
        return 0.1 * (x + 1) ** 2


f2 = lambda x: 0 if x <= 1 else 0.1 * (x - 1) ** 2

xpos = np.linspace(0, 10, 100, endpoint=True)
xneg = np.linspace(-10, 0, 100, endpoint=True)

ypos = [f1(i) for i in xpos]
ynegflat = [f2(i) for i in xneg]
ynegquad = [f1(i) for i in xneg]

plt.figure(figsize=(2, 2))
plt.scatter(0, 0, color="orange", s=30)
plt.plot(xpos, ypos, color="dodgerblue")
plt.plot(xneg, ynegquad, color="red")
plt.plot(xneg, ynegflat, color="green")
plt.xlabel(r"$\mu$")
plt.ylabel(r"$L_\mathcal{S}(\bm{w}^* + \mu\bm{\hat{u}})$")
plt.xticks([])
plt.yticks([])
plt.savefig("asymmetry-plots/asymmetry-fake.pdf")
