"""
plot_maml_vs_baseline.py

Comparativa a 3 vias sobre la fase 2 del cubo:
    - QR-DQN fine-tune       (continual learning clasico, off-policy)
    - PPO desde cero         (baseline justo, on-policy sin transferencia)
    - MAML/Reptile few-shot  (meta-learning + adaptacion rapida)

Normaliza nombres de columnas (episode, timestep, length, reward) y
recorta el eje X al rango comun para que se vea quien aprende mas
rapido en los primeros pasos. Tambien genera una vista 'completa' por
si se quiere observar el QR-DQN hasta el final.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Mapeo flexible: nombre canonico -> alternativas posibles en CSV
COL_MAP = {
    "episode":  ["episode", "episodio"],
    "timestep": ["timestep", "timesteps", "step", "steps"],
    "reward":   ["reward", "recompensa", "ep_reward"],
    "length":   ["length", "ep_length", "longitud", "time_alive_steps"],
}

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


def plot_curva(ax, df, label, color, x_key="timestep", y_key="length",
               mm_w=20, alpha_raw=0.18):
    if x_key not in df or y_key not in df:
        return
    x, y = df[x_key].values, df[y_key].values
    ax.plot(x, y, color=color, alpha=alpha_raw, lw=0.8)
    mm = media_movil(y, w=mm_w)
    if len(mm) < len(x):
        x_mm = x[len(x) - len(mm):]
    else:
        x_mm = x
    ax.plot(x_mm, mm, color=color, lw=2.2, label=label)


def stats(df, name):
    n = len(df)
    rmean = df["reward"].mean() if "reward" in df else float("nan")
    rmax = df["reward"].max() if "reward" in df else float("nan")
    lmean = df["length"].mean() if "length" in df else float("nan")
    lmax = df["length"].max() if "length" in df else float("nan")
    tmax = df["timestep"].max() if "timestep" in df else float("nan")
    return (f"{name:<22} eps={n:5d}  T={tmax:>8.0f}  "
            f"r_med={rmean:7.2f}  r_max={rmax:7.2f}  "
            f"l_med={lmean:6.1f}  l_max={lmax:5.0f}")


def episodios_hasta(df, target):
    if "length" not in df:
        return None
    idx = df.index[df["length"] >= target]
    return int(df.loc[idx[0], "episode"]) if len(idx) else None


def timesteps_hasta(df, target):
    if "length" not in df or "timestep" not in df:
        return None
    idx = df.index[df["length"] >= target]
    return int(df.loc[idx[0], "timestep"]) if len(idx) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrdqn",    default="metrics/qrdqn_finetune_cubo2_metrics.csv")
    ap.add_argument("--baseline", default="metrics/baseline_scratch_cubo2.csv")
    ap.add_argument("--maml",     default="metrics/maml_fewshot_cubo2.csv")
    ap.add_argument("--out_prefix", default="metrics/comparativa_3vias")
    ap.add_argument("--zoom_steps", type=int, default=10000,
                    help="X-max para la vista zoom (timesteps)")
    args = ap.parse_args()

    fuentes = [
        ("QR-DQN fine-tune", args.qrdqn,    "#d62728"),  # rojo
        ("PPO scratch",      args.baseline, "#888888"),  # gris
        ("MAML few-shot",    args.maml,     "#1f77b4"),  # azul
    ]

    dfs = {}
    for name, path, _ in fuentes:
        if not Path(path).exists():
            print(f"[AVISO] no existe {path}; se omite '{name}'")
            continue
        dfs[name] = normaliza(pd.read_csv(path))

    if not dfs:
        print("[ERROR] ningun CSV cargado"); return

    # --- Trunca todas las series al rango comun para una comparacion justa ---
    # El budget comun = max timestep del experimento mas corto (tipicamente
    # MAML o PPO scratch con args.adapt_steps). Truncar QR-DQN aqui evita
    # comparar 80k vs 8k.
    if any("timestep" in d for d in dfs.values()):
        budget_comun = min(d["timestep"].max() for d in dfs.values()
                           if "timestep" in d)
        print(f"[truncado] budget comun = {budget_comun:.0f} timesteps")
        for name in list(dfs.keys()):
            d = dfs[name]
            if "timestep" in d:
                dfs[name] = d[d["timestep"] <= budget_comun].reset_index(drop=True)

    # ---- Figura 1: longitud vs timesteps (vista completa) ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, _, color in fuentes:
        if name in dfs:
            plot_curva(ax, dfs[name], name, color, "timestep", "length")
    ax.set_xlabel("Timesteps en cubo 2")
    ax.set_ylabel("Longitud de episodio (steps)")
    ax.set_title("Velocidad de aprendizaje en cubo 2 - vista completa")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    p1 = f"{args.out_prefix}_completo.png"
    fig.tight_layout(); fig.savefig(p1, dpi=140); plt.close(fig)

    # ---- Figura 2: zoom a primeros adapt_steps timesteps ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, _, color in fuentes:
        if name in dfs:
            sub = dfs[name][dfs[name]["timestep"] <= args.zoom_steps]
            if len(sub) > 0:
                plot_curva(ax, sub, name, color, "timestep", "length", mm_w=8)
    ax.set_xlabel(f"Timesteps en cubo 2 (zoom <= {args.zoom_steps})")
    ax.set_ylabel("Longitud de episodio (steps)")
    ax.set_title("Velocidad de aprendizaje - primeros pasos")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    p2 = f"{args.out_prefix}_zoom.png"
    fig.tight_layout(); fig.savefig(p2, dpi=140); plt.close(fig)

    # ---- Figura 3: best-so-far (envolvente) ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, _, color in fuentes:
        if name in dfs and "length" in dfs[name]:
            df = dfs[name]
            ax.plot(df["timestep"], df["length"].cummax(),
                    color=color, lw=2.2, label=name)
    ax.set_xlabel("Timesteps en cubo 2")
    ax.set_ylabel("Mejor longitud alcanzada")
    ax.set_title("Progreso real (best-so-far) en cubo 2")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    p3 = f"{args.out_prefix}_progreso.png"
    fig.tight_layout(); fig.savefig(p3, dpi=140); plt.close(fig)

    # ---- Resumen numerico ----
    print("\n=== Resumen ===")
    for name, _, _ in fuentes:
        if name in dfs:
            print(stats(dfs[name], name))

    # Metrica clave: cuantos timesteps necesita cada uno para alcanzar
    # un % de la mejor longitud global (mas justo que comparar episodios).
    long_max_global = max(
        dfs[n]["length"].max() for n in dfs if "length" in dfs[n]
    )
    for pct in (0.5, 0.8):
        target = long_max_global * pct
        print(f"\nTimesteps para alcanzar long >= {target:.0f} "
              f"({int(pct*100)}% del max global = {long_max_global:.0f}):")
        for name, _, _ in fuentes:
            if name in dfs:
                t = timesteps_hasta(dfs[name], target)
                print(f"  {name:<22} {t if t else 'no alcanza'}")

    print(f"\nFiguras: {p1}\n         {p2}\n         {p3}")


if __name__ == "__main__":
    main()
