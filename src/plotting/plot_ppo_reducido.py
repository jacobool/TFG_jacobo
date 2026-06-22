import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("metrics/ppo_metrics.csv")
window = 30

panels = [
    ("time_alive", "Tiempo de supervivencia (s)", "tab:blue"),
    ("reward",     "Recompensa total",            "tab:green"),
    ("train/entropy_loss", "Entropy loss",         "tab:purple"),
]

fig, axs = plt.subplots(len(panels), 1, figsize=(12, 3.2 * len(panels)))

for ax, (col, ylabel, color) in zip(axs, panels):
    serie = pd.to_numeric(df[col], errors="coerce")
    ep = df["episode"]

    ax.plot(ep, serie, color=color, alpha=0.25, linewidth=0.8)

    smooth = serie.rolling(window=window, min_periods=1).mean()
    ax.plot(ep, smooth, color=color, linewidth=2.2,
            label=f"Media {window} ep")

    if col in ("time_alive", "reward"):
        ax.axhline(y=serie.max(), color="gold", linestyle="--", alpha=0.6,
                   label=f"Máx: {serie.max():.2f}")

    ax.set_title(ylabel)
    ax.set_xlabel("Episodio")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("metrics/ppo_plot_reducido.png", dpi=120)
plt.close(fig)
print("Guardado en metrics/ppo_plot_reducido.png")
