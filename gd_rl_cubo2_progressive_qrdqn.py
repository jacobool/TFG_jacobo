"""Progressive Networks sobre QR-DQN: cubo 1 -> cubo 2.

Replica del esquema de gd_rl_cubo2_progressive.py, pero usando QR-DQN
como motor base en lugar de DQN clasico. Asi cierra la asimetria
algoritmica del estudio comparativo: TODAS las ramas (scratch,
fine-tune, distill, replay, EWC y ahora PNN) operan sobre QR-DQN y
parten del mismo profesor gd_qrdqn_440000_steps.zip.

Estructura:
  - Reutilizamos ProgressiveCNN desde el script original (mismo fix
    de inicializacion no-cero ya aplicado).
  - Cambia: la clase del modelo (QRDQN), la carga del profesor del
    cubo 1 (de QRDQN.load en lugar de DQN.load) y la inyeccion de
    pesos en quantile_net + quantile_net_target en lugar de
    q_net + q_net_target. La cabeza Q se reinicializa de cero (es
    el comportamiento por defecto y la limitacion teorica de PNN
    que documenta la memoria).

ANTES DE EJECUTAR:
  - Posiciona al jugador al INICIO de la segunda parte del cubo
    (Start Position en el editor del cubo 2).
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th

from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Reutilizamos el extractor con el fix de inicializacion.
from gd_rl_cubo2_progressive import ProgressiveCNN, GDEnvCubo, FEATURES_DIM


# ---------------------------------------------------------------- Config
METRICS_DIR = "metrics"
RUN_NAME    = "qrdqn_progressive_cubo2"

# Profesor QR-DQN del cubo 1 (mismo que usan scratch, fine-tune, distill,
# replay y EWC sobre QR-DQN).
MODELO_CUBO1 = "modelos_guardados/gd_qrdqn_440000_steps.zip"

TOTAL_STEPS = 600_000
SAVE_FREQ   = 25_000

# Hiperparametros alineados con la version DQN para que la unica
# variable cambiada sea el algoritmo base (DQN -> QR-DQN).
PNN_LEARNING_RATE        = 5e-5
PNN_EXPLORATION_INITIAL  = 1.0
PNN_EXPLORATION_FINAL    = 0.02
PNN_EXPLORATION_FRACTION = 0.15
PNN_LEARNING_STARTS      = 5_000
PNN_BUFFER_SIZE          = 50_000
PNN_TARGET_UPDATE        = 2_000
PNN_BATCH_SIZE           = 64
PNN_TRAIN_FREQ           = 4

QRDQN_LOG_KEYS = [
    "train/loss",
    "train/learning_rate",
    "rollout/exploration_rate",
]


# ----------------------------------------------------------- Plot helper
def save_plots(history, show=False):
    if not history:
        return
    df = pd.DataFrame(history)
    window = min(30, len(df))

    panels = [
        ("time_alive",                "Tiempo de supervivencia (s)",     "tab:blue"),
        ("ep_length",                 "Longitud de episodio (steps)",    "tab:cyan"),
        ("reward",                    "Recompensa total",                "tab:green"),
        ("train/loss",                "Quantile loss",                   "tab:red"),
        ("rollout/exploration_rate",  "Epsilon (exploracion)",           "tab:purple"),
    ]
    panels = [p for p in panels if p[0] in df.columns]

    n = len(panels)
    fig, axs = plt.subplots(n + 1, 1, figsize=(12, 3.2 * (n + 1)))
    if n + 1 == 1:
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

    # Panel adicional: las tres escalas laterales (mismo formato que en
    # la version DQN, para verificar que el gradiente fluye).
    ax_lat = axs[-1]
    if {"lat_scale_2", "lat_scale_3", "lat_scale_fc"}.issubset(df.columns):
        ax_lat.plot(df["episode"], df["lat_scale_2"], color="tab:purple",
                    alpha=0.6, linewidth=1.3, label="conv2 (lat_scale_2)")
        ax_lat.plot(df["episode"], df["lat_scale_3"], color="tab:orange",
                    alpha=0.6, linewidth=1.3, label="conv3 (lat_scale_3)")
        ax_lat.plot(df["episode"], df["lat_scale_fc"], color="tab:brown",
                    alpha=0.6, linewidth=1.3, label="fc (lat_scale_fc)")
        ax_lat.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax_lat.set_title("Escalas laterales (intensidad col1 -> col2)")
        ax_lat.set_ylabel("Valor del escalar")
        ax_lat.set_xlabel("Episodio")
        ax_lat.legend(loc="best", fontsize=9)
        ax_lat.grid(alpha=0.3)
    else:
        ax_lat.set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(METRICS_DIR, f"{RUN_NAME}_plot.png")
    csv_path  = os.path.join(METRICS_DIR, f"{RUN_NAME}_metrics.csv")
    plt.savefig(plot_path, dpi=120)
    df.to_csv(csv_path, index=False)
    if show:
        plt.show()
    plt.close(fig)


class MetricsAndSaveCallback(BaseCallback):
    """Callback con captura por episodio + checkpoint + escalas laterales.

    Diferencia clave respecto a la version DQN: extraemos
    self.model.policy.quantile_net.features_extractor (no q_net).
    """

    def __init__(self, save_freq, save_path, run_name, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.run_name  = run_name
        self.history   = []
        self.t0        = time.time()
        self.ep_reward = 0.0
        self.ep_length = 0
        self.ep_start  = time.perf_counter()
        self.best_reward = -np.inf
        self.best_length = 0

    def _get_lateral_scales(self):
        """Lee los tres escalares aprendibles del extractor activo."""
        try:
            extractor = self.model.policy.quantile_net.features_extractor
            return {
                "lat_scale_2":  extractor.lateral_scale_2.item(),
                "lat_scale_3":  extractor.lateral_scale_3.item(),
                "lat_scale_fc": extractor.lateral_scale_fc.item(),
            }
        except AttributeError:
            return {"lat_scale_2": np.nan,
                    "lat_scale_3": np.nan,
                    "lat_scale_fc": np.nan}

    def _on_step(self):
        rewards = self.locals["rewards"]
        dones   = self.locals["dones"]

        self.ep_reward += float(rewards[0])
        self.ep_length += 1

        if dones[0]:
            ep = len(self.history) + 1
            time_alive = time.perf_counter() - self.ep_start
            self.best_reward = max(self.best_reward, self.ep_reward)
            self.best_length = max(self.best_length, self.ep_length)

            scales = self._get_lateral_scales()

            entry = {
                "episode": ep,
                "timesteps": self.num_timesteps,
                "time_alive": time_alive,
                "ep_length": self.ep_length,
                "reward": self.ep_reward,
                "best_reward": self.best_reward,
                "best_ep_length": self.best_length,
                **scales,
            }
            for key in QRDQN_LOG_KEYS:
                entry[key] = self.logger.name_to_value.get(key, np.nan)
            self.history.append(entry)

            print(f"[ep {ep:>4}] ts={self.num_timesteps:>6} | "
                  f"time={time_alive:>6.2f}s | reward={self.ep_reward:>+7.2f} | "
                  f"len={self.ep_length:>4} | "
                  f"lat=({scales['lat_scale_2']:+.4f}, "
                  f"{scales['lat_scale_3']:+.4f}, "
                  f"{scales['lat_scale_fc']:+.4f})")

            self.ep_reward = 0.0
            self.ep_length = 0
            self.ep_start  = time.perf_counter()

        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            ckpt = f"{self.save_path}_{self.num_timesteps}_steps.zip"
            self.model.save(ckpt)
            save_plots(self.history, show=False)
            print(f"  >> Checkpoint guardado: {ckpt}")

        return True


# ----------------------------------------------------------- Main
def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs("modelos_guardados", exist_ok=True)

    print("== PNN sobre QR-DQN: cubo 1 -> cubo 2 ==")
    print(f"   Profesor    : {MODELO_CUBO1}")
    print(f"   Pasos       : {TOTAL_STEPS:,}")
    print(f"   Buffer      : {PNN_BUFFER_SIZE:,}")
    print(f"   LR          : {PNN_LEARNING_RATE}")
    print()

    env = DummyVecEnv([lambda: GDEnvCubo()])
    env = VecFrameStack(env, n_stack=4)

    # ---- 1) Cargar profesor QR-DQN del cubo 1 -----------------------
    print(f"Cargando profesor {MODELO_CUBO1}...")
    cube1_model = QRDQN.load(MODELO_CUBO1, env=env, device="auto")

    # En QR-DQN el features_extractor cuelga de quantile_net (no q_net).
    cube1_cnn_state = {
        k: v.clone()
        for k, v in cube1_model.policy.quantile_net.features_extractor.state_dict().items()
    }

    del cube1_model
    if th.cuda.is_available():
        th.cuda.empty_cache()
    print(f"  Pesos del cubo 1 extraidos ({len(cube1_cnn_state)} tensores).\n")

    # ---- 2) Crear QRDQN con ProgressiveCNN ---------------------------
    print("Construyendo red progresiva sobre QR-DQN...")
    model = QRDQN(
        "CnnPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=ProgressiveCNN,
            features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
        ),
        verbose=1,
        buffer_size=PNN_BUFFER_SIZE,
        learning_starts=PNN_LEARNING_STARTS,
        batch_size=PNN_BATCH_SIZE,
        train_freq=PNN_TRAIN_FREQ,
        gradient_steps=1,
        target_update_interval=PNN_TARGET_UPDATE,
        learning_rate=PNN_LEARNING_RATE,
        gamma=0.99,
        exploration_fraction=PNN_EXPLORATION_FRACTION,
        exploration_initial_eps=PNN_EXPLORATION_INITIAL,
        exploration_final_eps=PNN_EXPLORATION_FINAL,
        device="auto",
    )

    # ---- 3) Inyectar pesos del cubo 1 en ambas columnas 1 -----------
    # Tanto en quantile_net como en quantile_net_target. La columna 2
    # se inicializa por copia para arrancar comportandose igual.
    for qnet in [model.quantile_net, model.quantile_net_target]:
        qnet.features_extractor.load_column1(cube1_cnn_state)
        qnet.features_extractor.init_column2_from_column1()

    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable    = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    frozen       = total_params - trainable
    print(f"  Parametros totales : {total_params:>10,}")
    print(f"  Entrenables (col2) : {trainable:>10,}")
    print(f"  Congelados  (col1) : {frozen:>10,}")

    # Recrear optimizador para que solo cubra parametros entrenables.
    trainable_params = [p for p in model.policy.parameters() if p.requires_grad]
    model.policy.optimizer = th.optim.Adam(trainable_params, lr=PNN_LEARNING_RATE)
    print(f"  Optimizador con {len(trainable_params)} grupos de parametros.\n")

    # ---- 4) Entrenar en cubo 2 ---------------------------------------
    print("Reposiciona el juego en CUBO 2 (Start Position 2a parte).")
    print(f"Se entrenaran {TOTAL_STEPS:,} pasos.")
    print(f"Las escalas laterales arrancan en 0.01 (fix de deadlock).")
    input("\nPulsa ENTER cuando GD este en cubo 2 y en primer plano...\n")

    callback = MetricsAndSaveCallback(
        save_freq=SAVE_FREQ,
        save_path=f"modelos_guardados/{RUN_NAME}",
        run_name=RUN_NAME,
    )

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callback,
        reset_num_timesteps=True,
        log_interval=10,
    )

    final_path = f"modelos_guardados/{RUN_NAME}_FINAL.zip"
    model.save(final_path)
    save_plots(callback.history, show=False)
    env.close()

    print()
    print(f"Modelo final  : {final_path}")
    print(f"Metricas      : {METRICS_DIR}/{RUN_NAME}_metrics.csv")
    print(f"Grafica       : {METRICS_DIR}/{RUN_NAME}_plot.png")


if __name__ == "__main__":
    main()
