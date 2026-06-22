import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

import time
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Importamos el entorno exactamente igual que en entrenamiento
from gd_rl_nave_2 import GDEnv, STEP_DURATION

# --- CONFIGURACIÓN DE EVALUACIÓN ---
CHECKPOINTS = {
    # "680k":  "modelos_guardados/nave_dqn_1_680000_steps",
    # "720k":  "modelos_guardados/nave_dqn_1_720000_steps",
    # "760k":  "modelos_guardados/nave_dqn_1_760000_steps",
    # "800k":  "modelos_guardados/nave_dqn_1_800000_steps",
    # "840k":  "modelos_guardados/nave_dqn_1_840000_steps",
    # "880k":  "modelos_guardados/nave_dqn_1_880000_steps",
    # "920k":  "modelos_guardados/nave_dqn_1_920000_steps",
    # "960k":  "modelos_guardados/nave_dqn_1_960000_steps",
    # "1000k": "modelos_guardados/nave_dqn_1_1000000_steps",
    "840k":  "modelos_guardados/nave_dqn_2_840000_steps",
    "880k":  "modelos_guardados/nave_dqn_2_880000_steps",
    "920k":  "modelos_guardados/nave_dqn_2_920000_steps",
    "960k":  "modelos_guardados/nave_dqn_2_960000_steps",
    "1000k": "modelos_guardados/nave_dqn_2_1000000_steps",
    "1040k": "modelos_guardados/nave_dqn_2_1040000_steps"
}

N_EPISODIOS = 15        # Episodios por checkpoint
EVAL_RESULTS_CSV = "eval_resultados_2.csv"
EVAL_PLOT_PNG    = "eval_comparativa_2.png"


# --- FUNCIÓN PRINCIPAL DE EVALUACIÓN ---
def evaluar_checkpoint(nombre, path, n_episodios):
    """
    Evalúa un checkpoint durante n_episodios y devuelve sus métricas.
    Condiciones idénticas al entrenamiento: mismo entorno, mismo timing.
    deterministic=True elimina la aleatoriedad de exploración.
    """
    # Verificar que el checkpoint existe antes de intentar cargarlo
    if not os.path.exists(path + ".zip"):
        print(f"  ⚠️  No encontrado: {path}.zip — omitiendo.")
        return None

    env = DummyVecEnv([lambda: GDEnv()])
    env = VecFrameStack(env, n_stack=4)
    model = DQN.load(path, env=env)

    tiempos = []
    obs = env.reset()

    for ep in range(n_episodios):
        done = False
        t_start = time.perf_counter()

        while not done:
            step_start = time.perf_counter()

            # deterministic=True: siempre la mejor acción conocida, sin exploración
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)

            # Timing idéntico al entrenamiento
            elapsed = time.perf_counter() - step_start
            sleep_time = STEP_DURATION - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        tiempo = time.perf_counter() - t_start
        tiempos.append(tiempo)
        print(f"    Ep {ep+1:02d}/{n_episodios}: {tiempo:.2f}s")

        obs = env.reset()

    env.close()

    return {
        "checkpoint": nombre,
        "media":      np.mean(tiempos),
        "mediana":    np.median(tiempos),
        "std":        np.std(tiempos),
        "mejor":      np.max(tiempos),
        "peor":       np.min(tiempos),
        "q25":        np.percentile(tiempos, 25),
        "q75":        np.percentile(tiempos, 75),
        "runs":       tiempos,
    }


# --- GRÁFICA COMPARATIVA ---
def guardar_grafica(resultados):
    """
    Genera dos gráficas:
    - Barras: mediana de supervivencia por checkpoint
    - Boxplot: distribución completa de cada checkpoint
    """
    nombres  = [r["checkpoint"] for r in resultados]
    medianas = [r["mediana"]    for r in resultados]
    medias   = [r["media"]      for r in resultados]
    stds     = [r["std"]        for r in resultados]
    runs     = [r["runs"]       for r in resultados]

    fig, axs = plt.subplots(2, 1, figsize=(14, 10))

    # --- Gráfica 1: Barras mediana + std ---
    x = np.arange(len(nombres))
    bars = axs[0].bar(x, medianas, color='steelblue', alpha=0.8, label='Mediana')
    axs[0].errorbar(x, medias, yerr=stds, fmt='o', color='darkred',
                    capsize=4, label='Media ± Std')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(nombres, rotation=45, ha='right')
    axs[0].set_title('Comparativa de Checkpoints — Tiempo de Supervivencia')
    axs[0].set_ylabel('Segundos')
    axs[0].legend()
    axs[0].grid(axis='y', alpha=0.3)

    # Marcar el mejor checkpoint por mediana
    mejor_idx = int(np.argmax(medianas))
    bars[mejor_idx].set_color('gold')
    bars[mejor_idx].set_edgecolor('darkorange')
    bars[mejor_idx].set_linewidth(2)
    axs[0].text(mejor_idx, medianas[mejor_idx] + 0.5,
                '★ MEJOR', ha='center', color='darkorange', fontweight='bold')

    # --- Gráfica 2: Boxplot distribución ---
    bp = axs[1].boxplot(runs, labels=nombres, patch_artist=True,
                        medianprops=dict(color='red', linewidth=2))
    for patch in bp['boxes']:
        patch.set_facecolor('lightsteelblue')
    bp['boxes'][mejor_idx].set_facecolor('gold')
    axs[1].set_xticklabels(nombres, rotation=45, ha='right')
    axs[1].set_title('Distribución por Checkpoint (Boxplot)')
    axs[1].set_ylabel('Segundos')
    axs[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(EVAL_PLOT_PNG, dpi=120)
    plt.show()
    print(f"\n📊 Gráfica guardada en '{EVAL_PLOT_PNG}'")


# --- MAIN ---
if __name__ == "__main__":
    print("="*55)
    print("   EVALUACIÓN DE CHECKPOINTS — MODO NAVE")
    print(f"   {N_EPISODIOS} episodios por checkpoint | deterministic=True")
    print("="*55)
    print("⚠️  Deja el juego en primer plano y no muevas el ratón.\n")

    resultados = []

    for nombre, path in CHECKPOINTS.items():
        print(f"\n🔍 Checkpoint {nombre}:")
        res = evaluar_checkpoint(nombre, path, N_EPISODIOS)
        if res is not None:
            resultados.append(res)

    if not resultados:
        print("❌ No se encontró ningún checkpoint. Revisa las rutas.")
        exit()

    # --- Tabla resumen en consola ---
    print("\n" + "="*65)
    print("  RESUMEN FINAL (ordenado por mediana)")
    print("="*65)
    print(f"{'Checkpoint':<10} {'Mediana':>9} {'Media':>9} {'Std':>8} {'Mejor':>8} {'Peor':>8}")
    print("-"*65)

    ordenados = sorted(resultados, key=lambda x: x["mediana"], reverse=True)
    for r in ordenados:
        marca = " ★" if r == ordenados[0] else ""
        print(f"{r['checkpoint']:<10} "
              f"{r['mediana']:>8.2f}s "
              f"{r['media']:>8.2f}s "
              f"{r['std']:>7.2f}s "
              f"{r['mejor']:>7.2f}s "
              f"{r['peor']:>7.2f}s"
              f"{marca}")

    # --- Guardar CSV ---
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "runs"} for r in resultados])
    df.to_csv(EVAL_RESULTS_CSV, index=False)
    print(f"\n💾 Resultados guardados en '{EVAL_RESULTS_CSV}'")

    # --- Gráfica ---
    guardar_grafica(resultados)

    # --- Recomendación final ---
    mejor = ordenados[0]
    mas_consistente = min(resultados, key=lambda x: x["std"])
    print("\n🏆 RECOMENDACIONES:")
    print(f"   Mayor mediana:     {mejor['checkpoint']} ({mejor['mediana']:.2f}s)")
    print(f"   Más consistente:   {mas_consistente['checkpoint']} (std={mas_consistente['std']:.2f}s)")

    if mejor["checkpoint"] != mas_consistente["checkpoint"]:
        print(f"\n   ⚠️  Son distintos. Si priorizas consistencia → {mas_consistente['checkpoint']}")
        print(f"        Si priorizas rendimiento pico           → {mejor['checkpoint']}")
    else:
        print(f"\n   ✅ Coinciden. Checkpoint recomendado: {mejor['checkpoint']}")