import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

"""
plot_comparativa_nivel2.py

Comparativa de 5 algoritmos sobre el NIVEL 2 (mapa nuevo) con presupuesto
comun de ~80k timesteps:

    - QR-DQN fine-tune    (transfer cubo1 -> nivel2, off-policy)
    - QR-DQN scratch      (off-policy desde cero)
    - PPO scratch         (on-policy desde cero, baseline justo de MAML)
    - MAML/Reptile        (meta-learning + adaptacion rapida)
    - PEARL adapt         (off-policy meta-RL + posterior sampling)

Genera 4 figuras:
    * comparativa_nivel2_curvas.png    -> longitud vs timesteps (raw+suavizado)
    * comparativa_nivel2_progreso.png  -> best-so-far (envolvente)
    * comparativa_nivel2_barras.png    -> resumen numerico (bar chart)
    * comparativa_nivel2_general.png   -> imagen unica con los 3 paneles
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COL_MAP = {
    "episode":  ["episode", "episodio"],
    "timestep": ["timestep", "timesteps", "step", "steps"],
    "reward":   ["reward", "recompensa", "ep_reward"],
    "length":   ["length", "ep_length", "longitud", "time_alive_steps"],
}

# (nombre legible, ruta CSV, color)
FUENTES = [
    ("QR-DQN fine-tune", "metrics/qrdqn_finetune_nivel2_metrics.csv", "#d62728"),
    ("QR-DQN scratch",   "metrics/qrdqn_scratch_nivel2_metrics.csv",  "#ff7f0e"),
    ("PPO scratch",      "metrics/baseline_scratch_nivel2.csv",       "#888888"),
    ("MAML few-shot",    "metrics/maml_fewshot_nivel2.csv",           "#1f77b4"),
    ("PEARL adapt",      "metrics/pearl_adapt_nivel2.csv",            "#2ca02c"),
]


def normaliza(df):
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame()
    for canon, alts in COL_MAP.items():
        for a in alts:
            if a.lower() in cols:
                out[canon] = df[cols[a.lower()]]
                break
    if "episode" not in out:
        out["episode"] = np.arange(1, len(df) + 1)
    return out


def media_movil(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_curva(ax, df, label, color, mm_w=25, alpha_raw=0.15):
    x, y = df["timestep"].values, df["length"].values
    ax.plot(x, y, color=color, alpha=alpha_raw, lw=0.7)
    mm = media_movil(y, w=mm_w)
    x_mm = x[len(x) - len(mm):] if len(mm) < len(x) else x
    ax.plot(x_mm, mm, color=color, lw=2.2, label=label)


def cargar_dfs():
    dfs = {}
    for name, path, _ in FUENTES:
        if not Path(path).exists():
            print(f"[AVISO] no existe {path}; se omite '{name}'")
            continue
        dfs[name] = normaliza(pd.read_csv(path))

    # Truncar al rango comun
    budget = min(d["timestep"].max() for d in dfs.values())
    print(f"[truncado] budget comun = {budget:.0f} timesteps")
    for name in list(dfs.keys()):
        dfs[name] = dfs[name][dfs[name]["timestep"] <= budget].reset_index(drop=True)
    return dfs, budget


def fig_curvas(dfs, out):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, _, color in FUENTES:
        if name in dfs:
            plot_curva(ax, dfs[name], name, color)
    ax.set_xlabel("Timesteps en nivel 2")
    ax.set_ylabel("Longitud de episodio (steps)")
    ax.set_title("Velocidad de aprendizaje en nivel 2 - 5 algoritmos")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_progreso(dfs, out):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, _, color in FUENTES:
        if name in dfs:
            d = dfs[name]
            ax.plot(d["timestep"], d["length"].cummax(),
                    color=color, lw=2.3, label=name)
    ax.set_xlabel("Timesteps en nivel 2")
    ax.set_ylabel("Mejor longitud alcanzada")
    ax.set_title("Progreso real (best-so-far) en nivel 2")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_barras(dfs, out):
    nombres = [n for n, _, _ in FUENTES if n in dfs]
    colores = [c for n, _, c in FUENTES if n in dfs]
    long_max = [dfs[n]["length"].max() for n in nombres]
    long_med = [dfs[n]["length"].mean() for n in nombres]
    rew_med  = [dfs[n]["reward"].mean() for n in nombres]
    eps      = [len(dfs[n]) for n in nombres]

    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
    metrics = [
        ("Longitud maxima (best-so-far final)", long_max, axs[0, 0]),
        ("Longitud media por episodio",          long_med, axs[0, 1]),
        ("Recompensa media por episodio",        rew_med,  axs[1, 0]),
        ("Numero de episodios completados",      eps,      axs[1, 1]),
    ]
    for title, vals, ax in metrics:
        bars = ax.bar(nombres, vals, color=colores, edgecolor="black", lw=0.6)
        ax.set_title(title)
        ax.set_xticks(range(len(nombres)))
        ax.set_xticklabels(nombres, rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:.1f}" if isinstance(v, float) else str(v),
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Resumen numerico - nivel 2 (budget comun)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_general(dfs, out):
    """Imagen unica con los 3 paneles apilados."""
    fig = plt.figure(figsize=(13, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.2])
    ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # panel 1: curvas
    for name, _, color in FUENTES:
        if name in dfs:
            plot_curva(ax1, dfs[name], name, color)
    ax1.set_title("Velocidad de aprendizaje (longitud vs timesteps)")
    ax1.set_xlabel("Timesteps en nivel 2")
    ax1.set_ylabel("Longitud (steps)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    # panel 2: progreso
    for name, _, color in FUENTES:
        if name in dfs:
            d = dfs[name]
            ax2.plot(d["timestep"], d["length"].cummax(),
                     color=color, lw=2.3, label=name)
    ax2.set_title("Progreso real (best-so-far)")
    ax2.set_xlabel("Timesteps en nivel 2")
    ax2.set_ylabel("Mejor longitud")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right", fontsize=9)

    # panel 3: barras (long_max)
    nombres = [n for n, _, _ in FUENTES if n in dfs]
    colores = [c for n, _, c in FUENTES if n in dfs]
    long_max = [dfs[n]["length"].max() for n in nombres]
    long_med = [dfs[n]["length"].mean() for n in nombres]
    width = 0.35
    x = np.arange(len(nombres))
    bars1 = ax3.bar(x - width/2, long_max, width, label="long_max",
                    color=colores, edgecolor="black", lw=0.6)
    bars2 = ax3.bar(x + width/2, long_med, width, label="long_med",
                    color=colores, edgecolor="black", lw=0.6, alpha=0.55)
    for b, v in zip(bars1, long_max):
        ax3.text(b.get_x() + b.get_width()/2, v, f"{v:.0f}",
                 ha="center", va="bottom", fontsize=8)
    for b, v in zip(bars2, long_med):
        ax3.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(nombres, rotation=15, ha="right")
    ax3.set_title("Resumen numerico (max y media de longitud por episodio)")
    ax3.grid(axis="y", alpha=0.3)
    ax3.legend()

    fig.suptitle("Comparativa general - 5 algoritmos en nivel 2", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def resumen_texto(dfs, budget):
    print("\n=== Resumen numerico nivel 2 (budget comun = "
          f"{budget:.0f} timesteps) ===")
    for name, _, _ in FUENTES:
        if name not in dfs:
            continue
        d = dfs[name]
        print(f"{name:<22} eps={len(d):5d}  "
              f"long_max={d['length'].max():5.0f}  "
              f"long_med={d['length'].mean():6.1f}  "
              f"rew_max={d['reward'].max():7.2f}  "
              f"rew_med={d['reward'].mean():6.2f}")

    # cuanto tarda cada uno en llegar al 50/80% del long_max global
    long_max_global = max(d["length"].max() for d in dfs.values())
    for pct in (0.5, 0.8):
        target = long_max_global * pct
        print(f"\nTimesteps para alcanzar long >= {target:.0f} "
              f"({int(pct*100)}% del max global = {long_max_global:.0f}):")
        for name, _, _ in FUENTES:
            if name in dfs:
                d = dfs[name]
                idx = d.index[d["length"] >= target]
                t = int(d.loc[idx[0], "timestep"]) if len(idx) else None
                print(f"  {name:<22} {t if t else 'no alcanza'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", default="metrics/comparativa_nivel2")
    args = ap.parse_args()

    dfs, budget = cargar_dfs()
    if not dfs:
        print("[ERROR] ningun CSV cargado"); return

    p1 = f"{args.out_prefix}_curvas.png"
    p2 = f"{args.out_prefix}_progreso.png"
    p3 = f"{args.out_prefix}_barras.png"
    p4 = f"{args.out_prefix}_general.png"

    fig_curvas(dfs, p1)
    fig_progreso(dfs, p2)
    fig_barras(dfs, p3)
    fig_general(dfs, p4)

    resumen_texto(dfs, budget)
    print(f"\nFiguras:\n  {p1}\n  {p2}\n  {p3}\n  {p4}")


if __name__ == "__main__":
    main()
