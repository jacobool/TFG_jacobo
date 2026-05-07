"""Filtra outliers de congelaciones (ventanas emergentes) del CSV de
RecurrentPPO y regenera el CSV limpio + las graficas en el mismo estilo
que `gd_rl_env_4_recurrentppo.py`.

Outlier = episodio cuyo segundos/step excede 3x la mediana del run.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")
RUN_NAME = "recurrentppo"
OUTLIER_FACTOR = 3.0
ROLLING_WINDOW = 30


def filter_outliers(df: pd.DataFrame, factor: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["s_per_step"] = df["time_alive"] / df["ep_length"].clip(lower=1)
    median_sps = df["s_per_step"].median()
    threshold = median_sps * factor
    mask = df["s_per_step"] <= threshold
    clean = df[mask].drop(columns=["s_per_step"]).reset_index(drop=True)
    outliers = df[~mask].drop(columns=["s_per_step"]).reset_index(drop=True)
    print(f"Mediana s/step: {median_sps:.4f}  |  umbral: {threshold:.4f}")
    print(f"Outliers descartados: {len(outliers)}/{len(df)} "
          f"({100*len(outliers)/len(df):.2f}%)")
    print(f"Tiempo en outliers: {outliers['time_alive'].sum():.0f}s "
          f"({outliers['time_alive'].sum()/60:.1f} min)")
    return clean, outliers


def save_plots(df: pd.DataFrame, run_name: str, suffix: str = "_clean"):
    window = min(ROLLING_WINDOW, len(df))
    panels = [
        ("time_alive", "Tiempo de supervivencia (s)", "tab:blue"),
        ("ep_length", "Longitud de episodio (steps)", "tab:cyan"),
        ("reward", "Recompensa total", "tab:green"),
        ("train/policy_gradient_loss", "Policy gradient loss", "tab:red"),
        ("train/value_loss", "Value loss", "tab:orange"),
        ("train/entropy_loss", "Entropy loss", "tab:purple"),
        ("train/approx_kl", "Approx KL", "tab:brown"),
        ("train/clip_fraction", "Clip fraction", "tab:pink"),
        ("train/explained_variance", "Explained variance", "tab:olive"),
    ]
    panels = [p for p in panels if p[0] in df.columns]

    n = len(panels)
    fig, axs = plt.subplots(n, 1, figsize=(12, 3.2 * n))
    if n == 1:
        axs = [axs]

    for ax, (col, title, color) in zip(axs, panels):
        serie = pd.to_numeric(df[col], errors="coerce")
        ax.plot(df["episode"], serie, color=color, alpha=0.25, linewidth=0.8)
        smooth = serie.rolling(window=window, min_periods=1).mean()
        ax.plot(df["episode"], smooth, color=color, linewidth=2.2,
                label=f"Media {window} ep")
        if col in ("time_alive", "reward", "ep_length"):
            ax.axhline(y=serie.max(), color="gold", linestyle="--", alpha=0.6,
                       label=f"Max: {serie.max():.2f}")
        ax.set_title(title)
        ax.set_xlabel("Episodio")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(METRICS_DIR, f"{run_name}{suffix}_plot.png")
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def main():
    csv_in = os.path.join(METRICS_DIR, f"{RUN_NAME}_metrics.csv")
    df = pd.read_csv(csv_in)
    print(f"Episodios originales: {len(df)} | timesteps: {df['timesteps'].iloc[-1]}")

    clean, outliers = filter_outliers(df, OUTLIER_FACTOR)

    csv_clean = os.path.join(METRICS_DIR, f"{RUN_NAME}_metrics_clean.csv")
    csv_outliers = os.path.join(METRICS_DIR, f"{RUN_NAME}_outliers.csv")
    clean.to_csv(csv_clean, index=False)
    outliers.to_csv(csv_outliers, index=False)

    plot_path = save_plots(clean, RUN_NAME, suffix="_clean")

    print()
    print("=== Resumen post-filtrado ===")
    print(f"Episodios limpios: {len(clean)}")
    print(f"Tiempo de supervivencia  -> media {clean['time_alive'].mean():.2f}s, "
          f"max {clean['time_alive'].max():.2f}s")
    print(f"Recompensa               -> media {clean['reward'].mean():+.3f}, "
          f"max {clean['reward'].max():.2f}")
    print(f"Longitud de episodio     -> media {clean['ep_length'].mean():.1f}, "
          f"max {int(clean['ep_length'].max())}")

    n = len(clean)
    if n >= 5:
        first = clean.iloc[: n // 5]
        last = clean.iloc[-n // 5:]
        print()
        print("Tendencia (primer vs ultimo quinto):")
        print(f"  time_alive: {first['time_alive'].mean():.2f}s -> "
              f"{last['time_alive'].mean():.2f}s")
        print(f"  reward    : {first['reward'].mean():+.3f} -> "
              f"{last['reward'].mean():+.3f}")

    print()
    print(f"CSV limpio    : {csv_clean}")
    print(f"CSV outliers  : {csv_outliers}")
    print(f"Grafica limpia: {plot_path}")


if __name__ == "__main__":
    main()
