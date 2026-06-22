import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

"""Indirect Transfer via Policy Distillation: cubo parte 1 -> cubo parte 2.

Implementacion del esquema de indirect transfer descrito en Wang et al.
(2023) seccion 5.2, basado en Rusu et al. (2016) "Policy Distillation".

  - Profesor (congelado): QR-DQN entrenado en la parte 1 del cubo.
  - Estudiante (nuevo, desde cero): QR-DQN aprendiendo la parte 2.
  - Loss = quantile_huber_loss + lambda * KL(teacher || student),
    donde la distribucion sobre acciones se obtiene aplicando softmax
    a la media de los cuantiles (Q-value por accion).
  - lambda decae linealmente de DISTILL_COEF a 0 en DISTILL_DECAY_STEPS,
    de modo que al final del entreno la senhal del profesor desaparece
    y el estudiante optimiza solo el objetivo de la nueva tarea.

Antes de ejecutar:
  - Posiciona al jugador al inicio de la PARTE 2 del cubo (igual que
    para el fine-tuning anterior).
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
import torch.nn.functional as F

from sb3_contrib import QRDQN
from sb3_contrib.qrdqn.qrdqn import quantile_huber_loss
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from gd_rl_env_4 import GDEnv

# ---------------------------------------------------------------- Config
METRICS_DIR = "metrics"
RUN_NAME = "qrdqn_distill_cubo2"

TEACHER_CHECKPOINT = "modelos_guardados/gd_qrdqn_440000_steps.zip"

TOTAL_TIMESTEPS = 80000
SAVE_FREQ = 5000

# Hiperparametros del estudiante. Como aprende desde cero, son cercanos
# a los del entreno original pero con menos buffer (es un tramo corto).
STUDENT_LR = 5e-5
STUDENT_BUFFER = 30000
STUDENT_LEARNING_STARTS = 1000
STUDENT_TARGET_UPDATE = 2000
STUDENT_EXP_FRACTION = 0.3
STUDENT_BATCH_SIZE = 32

# Hiperparametros de destilacion.
DISTILL_COEF = 1.0           # peso inicial del termino KL
DISTILL_TEMPERATURE = 1.0    # temperatura del softmax (1.0 = estandar)
DISTILL_DECAY_STEPS = 40000  # decae a 0 en la primera mitad del entreno

QRDQN_LOG_KEYS = [
    "train/loss",
    "train/distill_loss",
    "train/distill_coef",
    "train/learning_rate",
    "rollout/exploration_rate",
]


# --------------------------------------------------------- QRDQN Distill
class DistilledQRDQN(QRDQN):
    """QRDQN con un termino adicional de KL-divergence con un profesor."""

    def __init__(self, *args, teacher=None, distill_coef=1.0,
                 distill_temperature=1.0, distill_decay_steps=40000, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.distill_coef_init = distill_coef
        self.distill_temperature = distill_temperature
        self.distill_decay_steps = distill_decay_steps
        if self.teacher is not None:
            for p in self.teacher.policy.parameters():
                p.requires_grad = False
            self.teacher.policy.set_training_mode(False)

    def _current_distill_coef(self) -> float:
        if self.num_timesteps >= self.distill_decay_steps:
            return 0.0
        return self.distill_coef_init * (
            1.0 - self.num_timesteps / self.distill_decay_steps
        )

    def _excluded_save_params(self) -> list:
        # Excluir el profesor del guardado: contiene el env con mss
        # (no picklable). El estudiante se guarda solo, lo cual es
        # correcto: para inferencia o evaluacion no se necesita al
        # profesor.
        return super()._excluded_save_params() + [
            "teacher", "distill_coef_init",
            "distill_temperature", "distill_decay_steps",
        ]

    def save(self, path, exclude=None, include=None):
        # Override defensivo: el cloudpickle de SB3 a veces sigue
        # referencias y serializa el profesor aunque este en
        # _excluded_save_params. Aqui lo desconectamos temporalmente.
        teacher_backup = self.teacher
        self.teacher = None
        try:
            super().save(path, exclude=exclude, include=include)
        finally:
            self.teacher = teacher_backup

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses, distill_losses = [], []
        coef = 0.0

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with th.no_grad():
                # quantile_net output shape: (batch, n_quantiles, n_actions)
                next_quantiles = self.quantile_net_target(
                    replay_data.next_observations
                )
                next_greedy = (
                    next_quantiles.mean(dim=1, keepdim=True)
                    .argmax(dim=2, keepdim=True)
                )
                next_greedy = next_greedy.expand(batch_size, self.n_quantiles, 1)
                next_quantiles = next_quantiles.gather(
                    dim=2, index=next_greedy
                ).squeeze(dim=2)
                target_quantiles = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_quantiles
                )

            current_all = self.quantile_net(replay_data.observations)
            actions = (
                replay_data.actions[..., None]
                .long()
                .expand(batch_size, self.n_quantiles, 1)
            )
            current_quantiles = th.gather(
                current_all, dim=2, index=actions
            ).squeeze(dim=2)

            base_loss = quantile_huber_loss(
                current_quantiles, target_quantiles, sum_over_quantiles=True
            )

            # ---- Termino de destilacion ----------------------------------
            coef = self._current_distill_coef()
            if self.teacher is not None and coef > 0:
                with th.no_grad():
                    # mean over n_quantiles (dim=1) -> Q por accion: (batch, n_actions)
                    teacher_q = (
                        self.teacher.quantile_net(replay_data.observations)
                        .mean(dim=1)
                    )
                    teacher_logp = F.log_softmax(
                        teacher_q / self.distill_temperature, dim=-1
                    )

                student_q = current_all.mean(dim=1)
                student_logp = F.log_softmax(
                    student_q / self.distill_temperature, dim=-1
                )

                # KL(teacher || student) = sum p_t * (logp_t - logp_s)
                p_t = teacher_logp.exp()
                kl = (p_t * (teacher_logp - student_logp)).sum(dim=-1).mean()
                distill_loss = (self.distill_temperature ** 2) * kl
            else:
                distill_loss = th.tensor(0.0, device=base_loss.device)

            loss = base_loss + coef * distill_loss
            losses.append(float(base_loss.detach().cpu().item()))
            distill_losses.append(float(distill_loss.detach().cpu().item()))

            self.policy.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates,
                           exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/distill_loss", np.mean(distill_losses))
        self.logger.record("train/distill_coef", coef)


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
        ("train/distill_loss", "Distillation KL loss", "tab:orange"),
        ("train/distill_coef", "Distill coef (lambda)", "tab:brown"),
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
        logger_vals = (getattr(self.model.logger, "name_to_value", {})
                       if hasattr(self.model, "logger") else {})
        for k in self.log_keys:
            values[k] = float(logger_vals.get(k, np.nan))
        if hasattr(self.model, "exploration_rate"):
            values["rollout/exploration_rate"] = float(self.model.exploration_rate)
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
            print(f"[AUTO-SAVE] '{model_name}.zip' | metricas -> "
                  f"{METRICS_DIR}/ (paso {self.num_timesteps})")

        return True


# ----------------------------------------------------------------- Main
if __name__ == "__main__":
    os.makedirs("modelos_guardados", exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    if not os.path.exists(TEACHER_CHECKPOINT):
        raise FileNotFoundError(
            f"No existe el checkpoint del profesor: {TEACHER_CHECKPOINT}.\n"
            f"Ajusta TEACHER_CHECKPOINT al .zip que quieras usar."
        )

    env = DummyVecEnv([lambda: GDEnv()])
    env = VecFrameStack(env, n_stack=4)

    print(f"[DT] Cargando profesor: {TEACHER_CHECKPOINT}")
    teacher = QRDQN.load(TEACHER_CHECKPOINT, env=env)
    teacher.policy.set_training_mode(False)
    for p in teacher.policy.parameters():
        p.requires_grad = False

    print("[DT] Creando estudiante (desde cero)")
    student = DistilledQRDQN(
        "CnnPolicy",
        env,
        teacher=teacher,
        distill_coef=DISTILL_COEF,
        distill_temperature=DISTILL_TEMPERATURE,
        distill_decay_steps=DISTILL_DECAY_STEPS,
        verbose=0,
        buffer_size=STUDENT_BUFFER,
        learning_starts=STUDENT_LEARNING_STARTS,
        train_freq=4,
        target_update_interval=STUDENT_TARGET_UPDATE,
        learning_rate=STUDENT_LR,
        exploration_fraction=STUDENT_EXP_FRACTION,
        batch_size=STUDENT_BATCH_SIZE,
        gamma=0.99,
    )

    print(f"[DT] LR={STUDENT_LR}  buffer={STUDENT_BUFFER}  "
          f"distill_coef={DISTILL_COEF}  T={DISTILL_TEMPERATURE}  "
          f"decay={DISTILL_DECAY_STEPS}")
    print(f"[DT] Total timesteps: {TOTAL_TIMESTEPS}, save cada {SAVE_FREQ}")
    print(f"[DT] Metricas -> {METRICS_DIR}/{RUN_NAME}_metrics.csv, "
          f"{METRICS_DIR}/{RUN_NAME}_plot.png")
    print()
    print("Asegurate de que el jugador esta posicionado al inicio de la "
          "PARTE 2 del cubo antes de continuar...")
    for i in range(5, 0, -1):
        print(f"  empezando en {i}s...")
        time.sleep(1)

    callback = MetricsAndSaveCallback(
        save_freq=SAVE_FREQ,
        save_path=f"modelos_guardados/gd_{RUN_NAME}",
        run_name=RUN_NAME,
        log_keys=QRDQN_LOG_KEYS,
    )

    student.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        reset_num_timesteps=True,
    )

    student.save(f"modelos_guardados/gd_{RUN_NAME}_FINAL")
    save_plots(callback.history, RUN_NAME, show=True)
    env.close()
    print("[DT] Distillation training terminado.")
