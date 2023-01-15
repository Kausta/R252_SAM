import matplotlib.pyplot as plt

plt.style.use("seaborn")

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{bm}\renewcommand{\rmdefault}{ptm}\renewcommand{\sfdefault}{phv}",
        "font.family": "sans",
        "font.size": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "legend.fancybox": True,
        "figure.figsize": (5.4, 4.0),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0,
    }
)