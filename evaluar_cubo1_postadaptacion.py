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
from gd_rl_cubo2_progressive import ProgressiveCNN  # noqa: F401


# ---------------------------------------------------------------- Config
N_PARTIDAS_POR_MODELO = 10
OUTPUT_CSV = "metrics/eval_cubo1_postadapt.csv"

# Tope duro por partida en segundos. Si una partida lo supera es porque
# la deteccion de muerte fallo y se solaparon dos partidas, no porque
# el agente este pasandose el cubo 1. El maximo real con los pinchos
# colocados al final del cubo 1 es ~23s, asi que 24.5s da margen
# suficiente para los runs largos legitimos sin tragarse outliers.
MAX_PARTIDA_SECONDS = 24.5

# Lista de modelos a evaluar. Comenta los que no tengas todavia.
# (etiqueta_legible, ruta_zip, clase) — la clase se autodetecta por
# nombre pero la dejo explicita por claridad.
MODELS_TO_EVALUATE = [
    # ("scratch_qrdqn",   "models/gd_qrdqn_scratch_cubo2_FINAL.zip",   QRDQN),
    # ("finetune_qrdqn",  "models/gd_qrdqn_finetune_cubo2_FINAL.zip",  QRDQN),
    # ("distill_qrdqn",   "models/gd_qrdqn_distill_cubo2_FINAL.zip",   QRDQN),
    # ("replay_qrdqn",    "models/gd_qrdqn_replay_cubo2_FINAL.zip",    QRDQN),
    # ("ewc_qrdqn",       "models/gd_qrdqn_ewc_cubo2_FINAL.zip",       QRDQN),
    ("pnn_qrdqn",       "models/qrdqn_progressive_cubo2_FINAL.zip",  QRDQN),
    # ("ewc_dqn",         "models/cubo2_ewc2_FINAL.zip",                DQN),
    # ("pnn_dqn",         "models/cubo2_progressive_FINAL.zip",         DQN),
    # ("baseline_dqn",   "models/gd_dqn_FINAL_4.zip",                              DQN),  # Como referencia (cubo 1 sin adaptar): el agente original.
    # # Como referencia (cubo 1 sin adaptar): el agente original.
    # ("baseline_cubo1",  "models/gd_qrdqn_440000_steps.zip",          QRDQN),
]


# ----------------------------------------------------------- Helpers
def safe_load(AlgoCls, path, env):
    """Carga un modelo SB3 sorteando el bug del optimizador recortado.

    Los modelos PNN (gd_rl_cubo2_progressive*.py) reconstruyen el
    optimizador con solo los parametros entrenables (columna 2 +
    adapters + cabeza Q, dejando fuera la columna 1 congelada). Al
    cargarlos con AlgoCls.load(), SB3 instancia un optimizador nuevo
    sobre TODOS los parametros y al restaurar su state_dict los
    grupos no coinciden:

        RuntimeError: loaded state dict contains a parameter group
        that doesn't match the size of optimizer's group

    Para evaluacion no necesitamos el optimizador (solo .predict()),
    asi que reintentamos cargando los pesos sin tocar el optimizador.
    """
    try:
        return AlgoCls.load(path, env=env)
    except Exception as e:
        # Capturamos cualquier excepcion y discriminamos por el
        # mensaje, no por el tipo. PyTorch lanza ValueError pero
        # algunas versiones de SB3 lo re-envuelven y el tipo concreto
        # cambia entre versiones (RuntimeError, KeyError, etc.).
        if "parameter group" not in str(e):
            raise

        print(f"  [aviso] optimizador recortado detectado "
              f"({type(e).__name__}); "
              f"recargando sin restaurar el optimizador...")
        from stable_baselines3.common.save_util import (
            load_from_zip_file, recursive_setattr)
        from stable_baselines3.common.utils import get_device

        device = get_device("auto")
        data, params, pytorch_variables = load_from_zip_file(path, device=device)

        # Filtrar las claves de optimizador para que set_parameters no
        # intente restaurarlas. Los pesos de la red (policy) si se cargan.
        params_sin_opt = {k: v for k, v in params.items()
                          if "optimizer" not in k.lower()}

        # Construir modelo vacio con los settings guardados.
        model = AlgoCls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,
        )
        model.__dict__.update(data)
        model._setup_model()

        # Inyectar los pesos sin tocar el optimizador.
        model.set_parameters(params_sin_opt, exact_match=False,
                             device=device)

        if pytorch_variables is not None:
            for name in pytorch_variables:
                recursive_setattr(model, name, pytorch_variables[name])

        return model


def play_n_episodes(model, env, n_episodes, label):
    """Juega n_episodes en el cubo 1 con politica determinista.

    Devuelve lista de dicts con time_alive y reward por partida.

    NOTA: el env.reset() se hace UNA sola vez antes del bucle. Dentro
    del bucle no llamamos reset() porque DummyVecEnv ya hace auto-reset
    cuando step() devuelve done=True, y la obs que devuelve en ese
    paso ya es la del nuevo episodio. Asi 'Episodio X iniciado' que
    imprime GDEnv coincide 1:1 con las partidas reales (con un solo
    print residual al final del ultimo episodio, sin coste).
    """
    print(f"\n=== Evaluando '{label}' ({n_episodes} partidas) ===")
    results = []
    obs = env.reset()  # solo UNA vez, fuera del bucle
    for i in range(n_episodes):
        ep_reward = 0.0
        t0 = time.perf_counter()
        done_arr = np.array([False])
        is_outlier = False
        while not done_arr.any():
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = env.step(action)
            ep_reward += float(rewards[0])
            done_arr = dones
            # Tope duro: si la deteccion de muerte fallo y la partida
            # se solapa con la siguiente, cortamos a mano. El env se
            # resetea explicitamente abajo para arrancar limpios.
            if time.perf_counter() - t0 > MAX_PARTIDA_SECONDS:
                is_outlier = True
                break
        t_alive = time.perf_counter() - t0
        if is_outlier:
            # Forzamos reset porque no hubo done=True real. Al haberse
            # solapado dos partidas, el env arrancara limpio en la
            # siguiente iteracion.
            obs = env.reset()
            print(f"  partida {i+1:>2}/{n_episodes}: "
                  f"time={t_alive:>6.2f}s | reward={ep_reward:>+7.2f} "
                  f"  [OUTLIER >{MAX_PARTIDA_SECONDS:.0f}s, deteccion fallida]")
        else:
            # Caso normal: tras done=True, DummyVecEnv ya hizo
            # auto-reset y obs es la primera obs del proximo episodio.
            print(f"  partida {i+1:>2}/{n_episodes}: "
                  f"time={t_alive:>6.2f}s | reward={ep_reward:>+7.2f}")
        results.append({"episode": i + 1,
                        "time_alive": t_alive,
                        "reward": ep_reward,
                        "outlier": is_outlier})
    return results


def summarize(rows, threshold_seconds=22.0):
    """Estadisticas agregadas por modelo + tasa de supervivencia.

    Las partidas marcadas como outlier (deteccion de muerte fallida)
    se EXCLUYEN de las medias y de la tasa de supervivencia, pero se
    reportan en n_outliers para no perderlos de vista.
    """
    n_total    = len(rows)
    valid      = [r for r in rows if not r.get("outlier", False)]
    n_outliers = n_total - len(valid)

    if not valid:
        # Caso patologico: todas las partidas fueron outliers.
        return {
            "n":               n_total,
            "n_validos":       0,
            "n_outliers":      n_outliers,
            "mean_time":       float("nan"),
            "std_time":        float("nan"),
            "max_time":        float("nan"),
            "mean_reward":     float("nan"),
            "survival_rate":   float("nan"),
            "survival_thr_s":  threshold_seconds,
        }

    times      = [r["time_alive"] for r in valid]
    rewards    = [r["reward"]     for r in valid]
    survivals  = sum(1 for t in times if t >= threshold_seconds)

    return {
        "n":               n_total,
        "n_validos":       len(valid),
        "n_outliers":      n_outliers,
        "mean_time":       float(np.mean(times)),
        "std_time":        float(np.std(times)),
        "max_time":        float(np.max(times)),
        "mean_reward":     float(np.mean(rewards)),
        "survival_rate":   survivals / len(valid),
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
            model = safe_load(AlgoCls, path, env)
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
        n_validos = summary["n_validos"]
        n_out     = summary["n_outliers"]
        print(f"  -> sobre {n_validos} validas (descartados {n_out} outliers): "
              f"media {summary['mean_time']:.2f}s "
              f"+/- {summary['std_time']:.2f} | "
              f"survival>={summary['survival_thr_s']}s: "
              f"{summary['survival_rate']*100:.0f}% "
              f"({int(summary['survival_rate']*n_validos)}/{n_validos})")

        # Sin pausa entre modelos: el script corre del tiron. El env
        # ya hizo auto-reset tras la ultima partida del modelo actual,
        # asi que el siguiente modelo arranca con observacion fresca.

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
        cols = ["model", "n", "n_validos", "n_outliers",
                "mean_time", "max_time", "mean_reward", "survival_rate"]
        print(df_sum[cols].to_string(index=False))

    print()
    print(f"Datos por partida : {raw_path}")
    print(f"Resumen por modelo: {sum_path}")


if __name__ == "__main__":
    main()
