import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from gd_rl_env_4 import GDEnv

METRICS_DIR = "metrics"
RUN_NAME = "ppo"
SAVE_FREQ = 10000
TOTAL_TIMESTEPS = 500000

PPO_LOG_KEYS = [
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
    "train/learning_rate",
]


def save_plots(history, run_name, show=False):
    if not history:
        return
    df = pd.DataFrame(history)
    window = min(30, len(df))

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
        ax.plot(df["episode"], smooth, color=color, linewidth=2.2, label=f"Media {window} ep")
        if col in ("time_alive", "reward", "ep_length"):
            ax.axhline(y=serie.max(), color="gold", linestyle="--", alpha=0.6,
                       label=f"Máx: {serie.max():.2f}")
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
    def __init__(self, save_freq, save_path, run_name, log_keys, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.run_name = run_name
        self.log_keys = log_keys
        self.history = []
        self.episode_count = 0
        self.ep_reward = 0.0
        self.ep_length = 0
        self.ep_start_time = time.time()
        self.best_reward = -np.inf
        self.best_ep_length = 0

    def _current_log_values(self):
        values = {}
        logger_vals = getattr(self.model.logger, "name_to_value", {}) if hasattr(self.model, "logger") else {}
        for k in self.log_keys:
            values[k] = float(logger_vals.get(k, np.nan))
        return values

    def _on_step(self) -> bool:
        self.ep_reward += float(self.locals["rewards"][0])
        self.ep_length += 1

        if self.locals["dones"][0]:
            self.episode_count += 1
            time_alive = time.time() - self.ep_start_time
            self.best_reward = max(self.best_reward, self.ep_reward)
            self.best_ep_length = max(self.best_ep_length, self.ep_length)

            entry = {
                "episode": self.episode_count,
                "timesteps": int(self.num_timesteps),
                "time_alive": time_alive,
                "ep_length": self.ep_length,
                "reward": self.ep_reward,
                "best_reward": self.best_reward,
                "best_ep_length": self.best_ep_length,
            }
            entry.update(self._current_log_values())
            self.history.append(entry)

            self.ep_reward = 0.0
            self.ep_length = 0
            self.ep_start_time = time.time()

        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            model_name = f"{self.save_path}_{self.num_timesteps}_steps"
            self.model.save(model_name)
            save_plots(self.history, self.run_name, show=False)
            print(f"💾 [AUTO-SAVE] '{model_name}.zip' | métricas → {METRICS_DIR}/ (paso {self.num_timesteps})")

        return True


if __name__ == "__main__":
    os.makedirs("modelos_guardados", exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    env = DummyVecEnv([lambda: GDEnv()])
    env = VecFrameStack(env, n_stack=4)

    callback = MetricsAndSaveCallback(
        save_freq=SAVE_FREQ,
        save_path=f"modelos_guardados/gd_{RUN_NAME}",
        run_name=RUN_NAME,
        log_keys=PPO_LOG_KEYS,
    )

    # PPO con hiperparámetros estilo Atari adaptados a entorno single-env y 15 FPS.
    # n_steps=128 ≈ 8.5s de juego por rollout; batch_size 64; 4 epochs por update.
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        learning_rate=2.5e-4,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
    )

    print(f"🚀 Entrenamiento PPO — {TOTAL_TIMESTEPS} pasos, guardado cada {SAVE_FREQ}.")
    print(f"   Métricas → {METRICS_DIR}/{RUN_NAME}_metrics.csv, {METRICS_DIR}/{RUN_NAME}_plot.png")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    model.save(f"gd_{RUN_NAME}_FINAL_4")
    save_plots(callback.history, RUN_NAME, show=True)
    env.close()
    print("✅ PPO terminado.")
