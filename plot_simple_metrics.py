"""
plot_simple_metrics.py

Genera una figura multi-panel en el mismo estilo que los plots de QR-DQN
para CSVs con columnas episode,timestep,reward,length (con o sin task).

Uso:
    python plot_simple_metrics.py --csv metrics/pearl_metrics.csv --out metrics/pearl_plot.png
    python plot_simple_metrics.py --csv metrics/sac_cubo1_metrics.csv --out metrics/sac_cubo1_plot.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None,
                    help="titulo de la figura (default: nombre del CSV)")
    ap.add_argument("--window", type=int, default=30,
                    help="ventana de la media movil")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "episode" not in df.columns:
        df["episode"] = np.arange(1, len(df) + 1)
    window = min(args.window, max(1, len(df)))

    panels = [
        ("length", "Longitud de episodio (steps)", "tab:cyan"),
        ("reward", "Recompensa total", "tab:green"),
    ]
    # panel extra: best-so-far de longitud (envolvente, "progreso real")
    if "length" in df.columns:
        df["_best_length"] = df["length"].cummax()
        panels.append(("_best_length",
                       "Mejor longitud alcanzada (best-so-far)",
                       "tab:orange"))
    if "task" in df.columns and df["task"].nunique() > 1:
        panels.append(("task", "Tarea activa (id)", "tab:purple"))

    panels = [p for p in panels if p[0] in df.columns]

    n = len(panels)
    fig, axs = plt.subplots(n, 1, figsize=(12, 3.2 * n))
    if n == 1:
        axs = [axs]

    title_prefix = args.title or Path(args.csv).stem

    for ax, (col, ylabel, color) in zip(axs, panels):
        serie = pd.to_numeric(df[col], errors="coerce")
        # raw
        ax.plot(df["episode"], serie, color=color, alpha=0.25, linewidth=0.8)
        # smoothed (no aplicar a best-so-far ni task, ya son monotonos/discretos)
        if col not in ("_best_length", "task"):
            smooth = serie.rolling(window=window, min_periods=1).mean()
            ax.plot(df["episode"], smooth, color=color, linewidth=2.2,
                    label=f"Media {window} ep")
        else:
            ax.plot(df["episode"], serie, color=color, linewidth=2.0,
                    label=col.lstrip("_"))
        # max horizontal
        if col in ("length", "reward"):
            ax.axhline(y=serie.max(), color="gold", linestyle="--", alpha=0.6,
                       label=f"Max: {serie.max():.2f}")
        ax.set_title(f"{title_prefix} - {ylabel}")
        ax.set_xlabel("Episodio")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=120)
    plt.close(fig)

    print(f"[plot] {args.csv}  ->  {args.out}")
    print(f"  episodios={len(df)}  long_max={df['length'].max():.0f}  "
          f"reward_max={df['reward'].max():.2f}  "
          f"reward_med={df['reward'].mean():.2f}")


if __name__ == "__main__":
    main()
