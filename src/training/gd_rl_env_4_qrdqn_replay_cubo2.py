import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

"""Adaptacion al cubo 2 con replay buffer mixing (CLEAR-style).

Mismo planteamiento que gd_rl_env_4_qrdqn_finetune_cubo2.py, pero
con un truco adicional para evitar el olvido catastrofico: antes de
empezar a entrenar, se carga en el replay buffer un conjunto de
transiciones del cubo 1 (generadas por generar_replay_cubo1.py).

Como el buffer tiene capacidad amplia (200 000), las transiciones del
cubo 1 NO se expulsan durante los 80k pasos de entrenamiento sobre
cubo 2. Cada batch de gradiente combina muestras de los dos tramos,
por lo que el optimizador no puede 'olvidar' cubo 1 mientras aprende
cubo 2 - es la version software del muestreo multi-Start-Position.

ANTES DE EJECUTAR:
  - Asegurate de que existe models/replay_cubo1.pkl (genera_replay_cubo1.py).
  - Posiciona al jugador al INICIO de la segunda parte del cubo
    (Start Position en el editor del cubo 2 o nivel custom recortado).
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from gd_rl_env_4 import GDEnv


# ----------------------------------------------------------- Helpers
def ensure_buffer_chw(buf, target_shape):
    """Si el buffer pickled estaba en HWC (84,84,4), lo transpone a CHW (4,84,84).

    Este caso aparece cuando el buffer fue generado por una version del
    script generar_replay_cubo1.py anterior al fix de VecTransposeImage.
    Detectamos el formato comparando la shape de las obs almacenadas con
    la observation_space que QRDQN espera (siempre CHW).
    """
    actual = buf.observations.shape[2:]   # quita (capacity, n_envs)
    if actual == target_shape:
        print("  [buffer] formato ya en CHW, no se necesita conversion.")
        return

    print(f"  [buffer] detectado HWC {actual} -> transponiendo a CHW "
          f"{target_shape}...")
    # Pasa de (cap, n_envs, H, W, C) a (cap, n_envs, C, H, W).
    buf.observations      = np.transpose(buf.observations,      (0, 1, 4, 2, 3))
    buf.next_observations = np.transpose(buf.next_observations, (0, 1, 4, 2, 3))
    # Reescribe la shape interna por consistencia con SB3.
    buf.obs_shape = target_shape
    print(f"  [buffer] conversion OK. Nueva shape obs: "
          f"{buf.observations.shape}")

# ---------------------------------------------------------------- Config
METRICS_DIR = "metrics"
RUN_NAME = "qrdqn_replay_cubo2_600k"

SOURCE_CHECKPOINT  = "modelos_guardados/gd_qrdqn_440000_steps.zip"
SOURCE_REPLAY_BUFF = "models/replay_cubo1.pkl"

# Presupuesto extendido para emparejar con EWC (~600k pasos). En runs
# largos el buffer FIFO empieza a evictar cubo 1 a partir del paso
# ~170k; lo mitigamos con re-inyeccion periodica (REINJECT_FREQ).
TOTAL_TIMESTEPS = 600_000
SAVE_FREQ       = 25_000

# Hiperparametros: identicos al fine-tune EXCEPTO buffer_size, mas grande
# para que las 30k transiciones precargadas del cubo 1 no se expulsen
# nada mas empezar; aun asi, por encima de ~170k pasos hace falta
# re-inyectar (ver REINJECT_FREQ).
RP_LEARNING_RATE        = 1e-5
RP_EXPLORATION_INITIAL  = 0.15
RP_EXPLORATION_FINAL    = 0.02
RP_EXPLORATION_FRACTION = 0.25
RP_LEARNING_STARTS      = 100        # ya tenemos 30k transiciones
RP_BUFFER_SIZE          = 200_000    # 200k: equilibrio RAM / cubo1 vivo
RP_TARGET_UPDATE        = 2000

# Re-inyeccion del cubo 1: cada REINJECT_FREQ pasos, anhadimos al buffer
# activo todas las transiciones del cubo 1 que extrajimos al inicio. Asi
# las cubo1 NO desaparecen aunque el FIFO las haya rotado fuera. Sin
# este mecanismo el experimento "buffer mixing" deja de mezclar a partir
# del paso ~170k y degenera en fine-tune ingenuo a 600k.
REINJECT_FREQ = 30_000

QRDQN_LOG_KEYS = [
    "train/loss",
    "train/learning_rate",
    "rollout/exploration_rate",
]


# ----------------------------------------------------------- Plot helper
def save_plots(history, run_name, show=False):
    if not history:
        return
    df = pd.DataFrame(history)
    window = min(30, len(df))

    panels = [
        ("time_alive", "Tiempo de supervivencia (s)", "tab:blue"),
        ("ep_length", "Longitud de episodio (steps)", "tab:cyan"),
        ("reward", "Recompensa total", "tab:green"),
        ("train/loss", "Quantile loss", "tab:red"),
        ("rollout/exploration_rate", "Epsilon (exploracion)", "tab:purple"),
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
    plot_path = os.path.join(METRICS_DIR, f"{run_name}_plot.png")
    csv_path = os.path.join(METRICS_DIR, f"{run_name}_metrics.csv")
    plt.savefig(plot_path, dpi=120)
    df.to_csv(csv_path, index=False)
    if show:
        plt.show()
    plt.close(fig)


class MetricsAndSaveCallback(BaseCallback):
    """Callback con tres responsabilidades:

    (1) Captura por episodio del time_alive medido directamente con
        time.perf_counter(); GDEnv NO devuelve esa clave en `info`, asi
        que la version anterior leia 0.0 sistematicamente. Este callback
        lo mide al estilo del script de fine-tune.

    (2) Persistencia periodica de checkpoint + plot + CSV.

    (3) Re-inyeccion periodica de las transiciones del cubo 1 en el
        replay buffer activo (cada REINJECT_FREQ pasos). Imprescindible
        para que el buffer mixing siga mezclando despues de los primeros
        170k pasos, momento en que el FIFO ya ha rotado las originales.
    """

    def __init__(self, save_freq, save_path, run_name,
                 cubo1_pool=None, reinject_freq=REINJECT_FREQ, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.run_name = run_name
        self.history = []
        self.t0 = time.time()
        self.ep_reward = 0.0
        self.ep_length = 0
        self.ep_start = time.perf_counter()
        self.best_reward = -np.inf
        self.best_length = 0
        self.best_time = 0.0

        # Pool inmutable de transiciones cubo 1 para re-inyeccion.
        # Cada elemento es un dict con obs, next_obs, action, reward,
        # done, infos (None) ya en formato compatible con buffer.add().
        self.cubo1_pool   = cubo1_pool
        self.reinject_freq = reinject_freq
        self.last_reinject = 0

    def _reinject_cubo1(self):
        """Anhade todas las transiciones del pool cubo 1 al buffer activo.

        buffer.add() respeta el orden FIFO y usa la posicion buf.pos
        actual; tras la inyeccion las transiciones cubo 1 vuelven a ser
        las mas 'recientes' del buffer y no se evictan hasta dar otra
        vuelta completa.
        """
        if self.cubo1_pool is None or not self.cubo1_pool:
            return
        buf = self.model.replay_buffer
        for tr in self.cubo1_pool:
            buf.add(
                obs=tr["obs"],
                next_obs=tr["next_obs"],
                action=tr["action"],
                reward=tr["reward"],
                done=tr["done"],
                infos=tr["infos"],
            )
        print(f"  >> Re-inyectadas {len(self.cubo1_pool):,} transiciones "
              f"del cubo 1 (buf.pos={buf.pos}, size={buf.size():,})")

    def _on_step(self):
        rewards = self.locals["rewards"]
        dones   = self.locals["dones"]

        self.ep_reward += float(rewards[0])
        self.ep_length += 1

        # ---- Re-inyeccion periodica del pool cubo 1 ----
        if (self.cubo1_pool is not None
                and self.num_timesteps - self.last_reinject >= self.reinject_freq
                and self.num_timesteps > 0):
            self._reinject_cubo1()
            self.last_reinject = self.num_timesteps

        # ---- Cierre de episodio ----
        if dones[0]:
            ep = len(self.history) + 1
            time_alive = time.perf_counter() - self.ep_start
            self.best_reward = max(self.best_reward, self.ep_reward)
            self.best_length = max(self.best_length, self.ep_length)
            self.best_time   = max(self.best_time,   time_alive)

            entry = {
                "episode": ep,
                "timesteps": self.num_timesteps,
                "time_alive": time_alive,
                "ep_length": self.ep_length,
                "reward": self.ep_reward,
                "best_reward": self.best_reward,
                "best_ep_length": self.best_length,
            }
            for key in QRDQN_LOG_KEYS:
                entry[key] = self.logger.name_to_value.get(key, np.nan)
            self.history.append(entry)

            print(f"[ep {ep:>4}] ts={self.num_timesteps:>6} | "
                  f"time={time_alive:>6.2f}s | reward={self.ep_reward:>+7.2f} | "
                  f"len={self.ep_length:>4}")

            self.ep_reward = 0.0
            self.ep_length = 0
            self.ep_start = time.perf_counter()

        # ---- Checkpoint periodico ----
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            ckpt = f"{self.save_path}_{self.num_timesteps}_steps.zip"
            self.model.save(ckpt)
            save_plots(self.history, self.run_name, show=False)
            print(f"  >> Checkpoint guardado: {ckpt}")

        return True


# ----------------------------------------------------------- Main
def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs("modelos_guardados", exist_ok=True)

    print("== Replay-buffer mixing: cubo 1 -> cubo 2 ==")
    print(f"   Checkpoint   : {SOURCE_CHECKPOINT}")
    print(f"   Replay cubo1 : {SOURCE_REPLAY_BUFF}")
    print(f"   Pasos        : {TOTAL_TIMESTEPS:,}")
    print(f"   Buffer total : {RP_BUFFER_SIZE:,}")
    print()

    if not os.path.isfile(SOURCE_REPLAY_BUFF):
        raise FileNotFoundError(
            f"No existe {SOURCE_REPLAY_BUFF}. "
            f"Ejecuta antes generar_replay_cubo1.py.")

    env = DummyVecEnv([lambda: GDEnv()])
    env = VecFrameStack(env, n_stack=4)

    print(f"Cargando modelo {SOURCE_CHECKPOINT}...")
    model = QRDQN.load(
        SOURCE_CHECKPOINT,
        env=env,
        device="auto",
        custom_objects={
            "learning_rate":           RP_LEARNING_RATE,
            "exploration_initial_eps": RP_EXPLORATION_INITIAL,
            "exploration_final_eps":   RP_EXPLORATION_FINAL,
            "exploration_fraction":    RP_EXPLORATION_FRACTION,
            "learning_starts":         RP_LEARNING_STARTS,
            "buffer_size":             RP_BUFFER_SIZE,
            "target_update_interval":  RP_TARGET_UPDATE,
        },
    )

    print(f"Cargando replay buffer del cubo 1: {SOURCE_REPLAY_BUFF}...")
    model.load_replay_buffer(SOURCE_REPLAY_BUFF)

    # Compatibilidad con buffers generados antes del fix de VecTransposeImage:
    # si el buffer guardado esta en HWC, lo convertimos a CHW al vuelo.
    ensure_buffer_chw(
        model.replay_buffer,
        target_shape=model.observation_space.shape,
    )

    n_cubo1 = model.replay_buffer.size()
    print(f"  Buffer poblado con {n_cubo1:,} transiciones "
          f"del cubo 1 (capacidad total: {model.replay_buffer.buffer_size:,}).")

    # ---- Extraccion del pool inmutable cubo 1 para re-inyeccion ----
    # En runs largos (>= ~170k pasos), el FIFO eviccionaria estas
    # transiciones; las copiamos aparte y las re-inyectamos cada
    # REINJECT_FREQ pasos via callback.
    buf = model.replay_buffer
    cubo1_pool = []
    for i in range(n_cubo1):
        cubo1_pool.append({
            "obs":      buf.observations[i].copy(),
            "next_obs": buf.next_observations[i].copy(),
            "action":   buf.actions[i].copy(),
            "reward":   buf.rewards[i].copy(),
            "done":     buf.dones[i].copy(),
            "infos":    [{}],
        })
    print(f"  Pool cubo 1 extraido: {len(cubo1_pool):,} transiciones "
          f"que se re-inyectaran cada {REINJECT_FREQ:,} pasos.")
    print()

    print("Pon Geometry Dash en cubo 2 (Start Position de la 2a parte) y NO toques.")
    print("Empieza el entrenamiento en 5 segundos...")
    time.sleep(5)

    callback = MetricsAndSaveCallback(
        save_freq=SAVE_FREQ,
        save_path=f"modelos_guardados/gd_{RUN_NAME}",
        run_name=RUN_NAME,
        cubo1_pool=cubo1_pool,
        reinject_freq=REINJECT_FREQ,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        reset_num_timesteps=True,
        log_interval=10,
    )

    final_path = f"modelos_guardados/gd_{RUN_NAME}_FINAL.zip"
    model.save(final_path)
    # Persistimos tambien el buffer final (util si se quiere continuar
    # entrenando en otra sesion sin perder lo aprendido).
    model.save_replay_buffer(f"models/replay_{RUN_NAME}_FINAL.pkl")
    save_plots(callback.history, RUN_NAME, show=False)
    env.close()

    print()
    print(f"Modelo final  : {final_path}")
    print(f"Buffer final  : models/replay_{RUN_NAME}_FINAL.pkl")
    print(f"Metricas      : {METRICS_DIR}/{RUN_NAME}_metrics.csv")
    print(f"Grafica       : {METRICS_DIR}/{RUN_NAME}_plot.png")


if __name__ == "__main__":
    main()
