import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

"""Genera la grafica comparativa de los experimentos en la PARTE 2
del cubo: scratch vs fine-tuning vs distillation vs EWC vs PNN.

Lee los CSVs de metricas, filtra outliers de congelacion (s/step > 3x
mediana del run cuando hay ep_length, si no por time_alive), suaviza con
media movil de 30 episodios y dibuja tiempo de supervivencia, longitud
de episodio (cuando esta disponible) y recompensa total en funcion de
los episodios (eje comun a todos los runs).

Tambien calcula la sample efficiency rho1 = sum(reward) / timesteps,
metrica del survey de Wang et al. (2023) eq. 7. Cuando un run no
registra timesteps, se aproxima como suma de ep_length o, en su
defecto, numero de episodios.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")
WINDOW = 30
OUTLIER_FACTOR = 3.0

RUNS = [
    ("qrdqn_scratch_cubo2",  "Scratch",                     "tab:gray"),
    ("qrdqn_finetune_cubo2", "Fine-tuning (parte 1)",       "tab:orange"),
    ("qrdqn_distill_cubo2",  "Indirect transfer (distill)", "tab:green"),
    ("cubo2_ewc",            "EWC",                         "tab:blue"),
    ("cubo2_progressive",    "PNN",                         "tab:red"),
]


def load_clean(run_name: str) -> pd.DataFrame | None:
    path = os.path.join(METRICS_DIR, f"{run_name}_metrics.csv")
    if not os.path.exists(path):
        print(f"[WARN] No existe {path}, salta.")
        return None
    df = pd.read_csv(path)

    if "ep_length" in df.columns:
        df["s_per_step"] = df["time_alive"] / df["ep_length"].clip(lower=1)
        threshold = df["s_per_step"].median() * OUTLIER_FACTOR
        clean = df[df["s_per_step"] <= threshold].reset_index(drop=True)
    else:
        # Sin ep_length: usamos time_alive directo como proxy de outlier.
        threshold = df["time_alive"].median() * OUTLIER_FACTOR
        clean = df[df["time_alive"] <= threshold].reset_index(drop=True)

    n_out = len(df) - len(clean)
    print(f"[{run_name}] episodios={len(df)} outliers={n_out} "
          f"({100*n_out/max(1,len(df)):.1f}%)")
    return clean


def main():
    runs = []
    for name, label, color in RUNS:
        df = load_clean(name)
        if df is None or len(df) < 5:
            continue
        runs.append((name, label, color, df))

    if not runs:
        print("No hay datos suficientes para comparar.")
        return

    fig, axs = plt.subplots(3, 1, figsize=(13, 11))

    metrics = [
        ("time_alive", "Tiempo de supervivencia (s)"),
        ("ep_length", "Longitud de episodio (steps)"),
        ("reward",    "Recompensa total"),
    ]

    for ax, (col, title) in zip(axs, metrics):
        for name, label, color, df in runs:
            if col not in df.columns:
                continue
            x = df["episode"].values if "episode" in df.columns else np.arange(len(df))
            y = df[col].rolling(window=WINDOW, min_periods=5).mean().values
            ax.plot(x, y, color=color, linewidth=2.0, label=label)
            y_raw = df[col].rolling(window=5, min_periods=1).mean().values
            ax.fill_between(x, y_raw, y, color=color, alpha=0.10)
        ax.set_title(title)
        ax.set_xlabel("Episodio")
        ax.set_ylabel(col)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(METRICS_DIR, "comparativa_cubo2(2).png")
    plt.savefig(out, dpi=120)
    plt.close(fig)

    # Tabla resumen + sample efficiency rho1
    print()
    print("=== Resumen final por experimento ===")
    print(f"{'Experimento':<32} {'time_avg':>9} {'time_max':>9} "
          f"{'rew_avg':>9} {'rew_max':>9} {'rho1':>10}")
    for name, label, color, df in runs:
        last = df.tail(min(100, len(df)))
        if "timesteps" in df.columns and df["timesteps"].iloc[-1] > 0:
            total_steps = df["timesteps"].iloc[-1]
        elif "ep_length" in df.columns:
            total_steps = df["ep_length"].sum()
        else:
            total_steps = len(df)
        rho1 = df["reward"].sum() / max(1, total_steps)
        print(f"{label:<32} "
              f"{last['time_alive'].mean():9.2f} "
              f"{last['time_alive'].max():9.2f} "
              f"{last['reward'].mean():+9.3f} "
              f"{last['reward'].max():9.2f} "
              f"{rho1:10.5f}")

    print()
    print(f"Grafica comparativa guardada en: {out}")


if __name__ == "__main__":
    main()
