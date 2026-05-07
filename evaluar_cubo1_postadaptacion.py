"""Evaluacion sobre el cubo 1 de los modelos finales del estudio
comparativo de adaptacion al cubo 2.

Para cada estrategia (scratch, fine-tune, distill, replay, EWC-DQN,
EWC-QR-DQN, PNN-DQN, PNN-QR-DQN) carga el modelo final, juega
N_PARTIDAS_POR_MODELO partidas en el cubo 1 con politica determinista
(deterministic=True), mide tiempo de supervivencia y recompensa, y
volcado a metrics/eval_cubo1_postadapt.csv.

Esta es la metrica que valida la narrativa de retencion / olvido
catastrofico: cuanto mejor sea cada estrategia preservando cubo 1,
mas alta sera la tasa de supervivencia esperada.

ANTES DE EJECUTAR:
  - GD en CUBO 1 (Start Position al inicio del nivel).
  - El script ira pidiendo confirmacion entre modelos para que tengas
    tiempo de respirar / cambiar de checkpoint.
"""

import os
import time
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from gd_rl_env_4 import GDEnv

# Si vas a evaluar el modelo PNN, descomenta para que pickle encuentre
# la clase ProgressiveCNN al deserializar.
# from gd_rl_cubo2_progressive import ProgressiveCNN  # noqa: F401


# ---------------------------------------------------------------- Config
N_PARTIDAS_POR_MODELO = 20
OUTPUT_CSV = "metrics/eval_cubo1_postadapt.csv"

# Lista de modelos a evaluar. Comenta los que no tengas todavia.
# (etiqueta_legible, ruta_zip, clase) — la clase se autodetecta por
# nombre pero la dejo explicita por claridad.
MODELS_TO_EVALUATE = [
    ("scratch_qrdqn",   "modelos_guardados/gd_qrdqn_scratch_cubo2_FINAL.zip",   QRDQN),
    ("finetune_qrdqn",  "modelos_guardados/gd_qrdqn_finetune_cubo2_FINAL.zip",  QRDQN),
    ("distill_qrdqn",   "modelos_guardados/gd_qrdqn_distill_cubo2_FINAL.zip",   QRDQN),
    ("replay_qrdqn",    "modelos_guardados/gd_qrdqn_replay_cubo2_FINAL.zip",    QRDQN),
    ("ewc_qrdqn",       "modelos_guardados/gd_qrdqn_ewc_cubo2_FINAL.zip",       QRDQN),
    ("pnn_qrdqn",       "modelos_guardados/qrdqn_progressive_cubo2_FINAL.zip",  QRDQN),
    ("ewc_dqn",         "modelos_guardados/cubo2_ewc2_FINAL.zip",                DQN),
    ("pnn_dqn",         "modelos_guardados/cubo2_progressive_FINAL.zip",         DQN),
    # Como referencia (cubo 1 sin adaptar): el agente original.
    ("baseline_cubo1",  "modelos_guardados/gd_qrdqn_440000_steps.zip",          QRDQN),
]


# ----------------------------------------------------------- Helpers
def play_n_episodes(model, env, n_episodes, label):
    """Juega n_episodes en el cubo 1 con politica determinista.

    Devuelve lista de dicts con time_alive y reward por partida.
    """
    print(f"\n=== Evaluando '{label}' ({n_episodes} partidas) ===")
    results = []
    for i in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        t0 = time.perf_counter()
        done_arr = np.array([False])
        while not done_arr.any():
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = env.step(action)
            ep_reward += float(rewards[0])
            done_arr = dones
        t_alive = time.perf_counter() - t0
        results.append({"episode": i + 1,
                        "time_alive": t_alive,
                        "reward": ep_reward})
        print(f"  partida {i+1:>2}/{n_episodes}: "
              f"time={t_alive:>6.2f}s | reward={ep_reward:>+7.2f}")
    return results


def summarize(rows, threshold_seconds=20.0):
    """Estadisticas agregadas por modelo + tasa de supervivencia."""
    times = [r["time_alive"] for r in rows]
    rewards = [r["reward"] for r in rows]
    survivals = sum(1 for t in times if t >= threshold_seconds)
    return {
        "n":               len(rows),
        "mean_time":       float(np.mean(times)),
        "std_time":        float(np.std(times)),
        "max_time":        float(np.max(times)),
        "mean_reward":     float(np.mean(rewards)),
        "survival_rate":   survivals / max(len(rows), 1),
        "survival_thr_s":  threshold_seconds,
    }


# ----------------------------------------------------------- Main
def main():
    os.makedirs("metrics", exist_ok=True)

    print("=" * 60)
    print("  EVALUACION CUBO 1 POST-ADAPTACION")
    print("=" * 60)
    print(f"  Modelos:   {len(MODELS_TO_EVALUATE)}")
    print(f"  Partidas:  {N_PARTIDAS_POR_MODELO} por modelo")
    print(f"  Salida:    {OUTPUT_CSV}")
    print()
    print("ASEGURATE de que GD esta en el CUBO 1 (Start Position al")
    print("inicio del nivel) antes de continuar.")
    input("\nPulsa ENTER cuando este listo...\n")

    env = DummyVecEnv([lambda: GDEnv()])
    env = VecFrameStack(env, n_stack=4)

    all_rows = []
    summaries = []

    for label, path, AlgoCls in MODELS_TO_EVALUATE:
        if not os.path.isfile(path):
            print(f"\n[SKIP] '{label}' -> {path} no existe todavia.")
            continue

        print(f"\nCargando '{label}' desde {path}...")
        try:
            model = AlgoCls.load(path, env=env)
        except Exception as e:
            print(f"[ERROR] no se pudo cargar {path}: {e}")
            continue

        rows = play_n_episodes(model, env, N_PARTIDAS_POR_MODELO, label)
        for r in rows:
            r["model"] = label
        all_rows.extend(rows)

        summary = summarize(rows)
        summary["model"] = label
        summaries.append(summary)
        print(f"  -> media: {summary['mean_time']:.2f}s "
              f"+/- {summary['std_time']:.2f} | "
              f"survival>={summary['survival_thr_s']}s: "
              f"{summary['survival_rate']*100:.0f}% "
              f"({int(summary['survival_rate']*N_PARTIDAS_POR_MODELO)}/{N_PARTIDAS_POR_MODELO})")

        # Pausa entre modelos para reposicionar el juego si hace falta.
        if label != MODELS_TO_EVALUATE[-1][0]:
            input("\nPulsa ENTER para continuar con el siguiente modelo...\n")

    env.close()

    # Persistencia.
    df_raw = pd.DataFrame(all_rows)
    df_sum = pd.DataFrame(summaries)

    raw_path = OUTPUT_CSV
    sum_path = OUTPUT_CSV.replace(".csv", "_resumen.csv")
    df_raw.to_csv(raw_path, index=False)
    df_sum.to_csv(sum_path, index=False)

    print()
    print("=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    if not df_sum.empty:
        cols = ["model", "n", "mean_time", "max_time", "mean_reward",
                "survival_rate"]
        print(df_sum[cols].to_string(index=False))

    print()
    print(f"Datos por partida : {raw_path}")
    print(f"Resumen por modelo: {sum_path}")


if __name__ == "__main__":
    main()
