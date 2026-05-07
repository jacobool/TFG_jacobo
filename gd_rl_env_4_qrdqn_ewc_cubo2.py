"""EWC sobre QR-DQN: cubo 1 -> cubo 2.

Adaptacion al algoritmo de referencia QR-DQN del esquema EWC ya
implementado para DQN clasico en gd_rl_cubo2_ewc.py. Asi, la
comparacion del estudio comparativo del cubo 2 queda equilibrada:
TODAS las ramas operan sobre el mismo motor (QR-DQN) y arrancan del
mismo profesor (gd_qrdqn_440000_steps.zip).

Diferencia tecnica clave respecto a la version DQN:
  - DQN expone una salida escalar Q(s, a) directamente.
  - QR-DQN expone cuantiles z_i(s, a). Para estimar la Fisher se toma
    la media de los cuantiles como proxy de Q(s, a) y se aplica
    log_softmax sobre las acciones, igual que en el script de
    destilacion.
  - La penalizacion EWC se aplica sobre los parametros de
    self.quantile_net (la red distribucional), no sobre q_net.

ANTES DE EJECUTAR:
  Fase 1 (Fisher):   GD en cubo 1, Start Position al inicio del nivel.
  Fase 2 (Entreno):  GD en cubo 2, Start Position de la 2a parte.

Si ya existe FISHER_PATH la fase 1 se omite automaticamente y la
sesion arranca directamente en la fase 2 (igual que la version DQN).
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
RUN_NAME    = "qrdqn_ewc_cubo2"

SOURCE_CHECKPOINT = "modelos_guardados/gd_qrdqn_440000_steps.zip"
FISHER_PATH       = "models/qrdqn_ewc_fisher.pt"

# Para emparejar con EWC sobre DQN (~600k) y replay (600k).
TOTAL_TIMESTEPS = 600_000
SAVE_FREQ       = 25_000

# Hiperparametros EWC. Mismos valores que la version DQN para que la
# comparacion DQN+EWC vs QRDQN+EWC sea limpia.
EWC_LAMBDA         = 5_000     # peso de la regularizacion Fisher
EWC_FISHER_SAMPLES = 500       # muestras en el calculo de la Fisher

# Hiperparametros de entrenamiento (alineados con fine-tune QR-DQN).
EWC_LEARNING_RATE        = 1e-5
EWC_EXPLORATION_INITIAL  = 0.15
EWC_EXPLORATION_FINAL    = 0.02
EWC_EXPLORATION_FRACTION = 0.25
EWC_LEARNING_STARTS      = 1000
EWC_BUFFER_SIZE          = 50_000
EWC_TARGET_UPDATE        = 2000

QRDQN_LOG_KEYS = [
    "train/loss",
    "train/ewc_penalty",
    "train/learning_rate",
    "rollout/exploration_rate",
]


# --------------------------------------------------------- EWC over QRDQN
class EWCQRDQN(QRDQN):
    """QR-DQN con regularizacion EWC aplicada sobre quantile_net."""

    def __init__(self, *args, ewc_lambda: float = EWC_LAMBDA, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda      = ewc_lambda
        self.ewc_fisher      = None  # dict {nombre_param: tensor F_i}
        self.ewc_star_params = None  # dict {nombre_param: tensor theta*_i}
        self._last_ewc_loss  = 0.0

    # ---- Estimacion de la diagonal de Fisher --------------------------
    def compute_fisher(self, vec_env, n_samples: int = EWC_FISHER_SAMPLES) -> None:
        """Estima la FIM diagonal sobre quantile_net.

        Para cada paso recogido en el cubo 1: quantile_net produce
        cuantiles, los promediamos por accion para obtener Q(s,a),
        aplicamos log_softmax como proxy de log pi(a|s) y propagamos
        el gradiente de log pi(a*|s) (a* = argmax Q). El cuadrado del
        gradiente sobre cada parametro se acumula y al final se divide
        entre el numero de muestras.
        """
        print(f"\n[EWC] Calculando Fisher sobre {n_samples} pasos del cubo 1...")
        self.policy.set_training_mode(False)

        fisher: dict[str, th.Tensor] = {
            n: th.zeros_like(p)
            for n, p in self.quantile_net.named_parameters()
            if p.requires_grad
        }

        obs = vec_env.reset()
        count = 0
        while count < n_samples:
            obs_th, _ = self.policy.obs_to_tensor(obs)

            # quantile_net: (batch, n_quantiles, n_actions)
            quantiles = self.quantile_net(obs_th)
            q_values  = quantiles.mean(dim=1)            # (batch, n_actions)
            log_probs = F.log_softmax(q_values, dim=-1)

            action = q_values.argmax(dim=-1)
            selected_log_prob = log_probs[
                th.arange(len(action)), action
            ].sum()

            self.policy.optimizer.zero_grad()
            selected_log_prob.backward()

            for n, p in self.quantile_net.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)

            count += 1
            obs, _, dones, _ = vec_env.step(action.cpu().numpy())
            if dones.any():
                obs = vec_env.reset()

            if count % 100 == 0:
                print(f"  Fisher: {count}/{n_samples} pasos...")

        for n in fisher:
            fisher[n] /= n_samples

        self.ewc_fisher = fisher
        self.ewc_star_params = {
            n: p.detach().clone()
            for n, p in self.quantile_net.named_parameters()
            if p.requires_grad
        }

        self.policy.set_training_mode(True)
        total = sum(f.numel() for f in fisher.values())
        mean_f = sum(f.mean().item() for f in fisher.values()) / len(fisher)
        print(f"[EWC] Fisher OK. Parametros: {total:,} | F media: {mean_f:.2e}")
        print(f"[EWC] lambda={self.ewc_lambda}  =>  penalizacion inicial estimada: "
              f"~{self.ewc_lambda * mean_f:.2e} por parametro\n")

    # ---- Persistencia de la Fisher -------------------------------------
    def save_fisher(self, path: str) -> None:
        if self.ewc_fisher is None:
            raise RuntimeError("Fisher no calculada todavia.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        th.save({
            "fisher":      self.ewc_fisher,
            "star_params": self.ewc_star_params,
            "lambda":      self.ewc_lambda,
        }, path)
        print(f"[EWC] Fisher persistida en {path}")

    def load_fisher(self, path: str) -> None:
        data = th.load(path, map_location=self.device, weights_only=False)
        self.ewc_fisher      = {n: t.to(self.device) for n, t in data["fisher"].items()}
        self.ewc_star_params = {n: t.to(self.device) for n, t in data["star_params"].items()}
        n_params = sum(f.numel() for f in self.ewc_fisher.values())
        print(f"[EWC] Fisher cargada de {path} ({n_params:,} parametros).")

    # ---- Train override con penalizacion EWC ---------------------------
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        base_losses, ewc_penalties = [], []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with th.no_grad():
                # next_quantiles: (batch, n_quantiles, n_actions)
                next_quantiles = self.quantile_net_target(replay_data.next_observations)
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

            # ---- Penalizacion EWC sobre quantile_net ----------------
            ewc_penalty = th.tensor(0.0, device=self.device)
            if self.ewc_fisher is not None:
                for n, p in self.quantile_net.named_parameters():
                    if n in self.ewc_fisher and n in self.ewc_star_params:
                        diff = p - self.ewc_star_params[n]
                        ewc_penalty = ewc_penalty + (self.ewc_fisher[n] * diff.pow(2)).sum()

            loss = base_loss + (self.ewc_lambda / 2.0) * ewc_penalty

            base_losses.append(float(base_loss.detach().cpu().item()))
            ewc_penalties.append(float(ewc_penalty.detach().cpu().item()))

            self.policy.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self._last_ewc_loss = float(np.mean(ewc_penalties))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(base_losses)))
        self.logger.record("train/ewc_penalty", self._last_ewc_loss)

    def _excluded_save_params(self) -> list:
        # Estos atributos no son picklables limpiamente y se guardan
        # aparte mediante save_fisher().
        return super()._excluded_save_params() + [
            "ewc_fisher", "ewc_star_params",
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
        ("train/ewc_penalty", "Penalizacion EWC", "tab:orange"),
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
    csv_path  = os.path.join(METRICS_DIR, f"{run_name}_metrics.csv")
    plt.savefig(plot_path, dpi=120)
    df.to_csv(csv_path, index=False)
    if show:
        plt.show()
    plt.close(fig)


class MetricsAndSaveCallback(BaseCallback):
    """Misma estructura que el callback de fine-tune; mide time_alive
    con time.perf_counter() para no depender de info[]."""

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
            self.ep_start  = time.perf_counter()

        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            ckpt = f"{self.save_path}_{self.num_timesteps}_steps.zip"
            self.model.save(ckpt)
            save_plots(self.history, self.run_name, show=False)
            print(f"  >> Checkpoint guardado: {ckpt}")

        return True


# ----------------------------------------------------------- Main
def build_env():
    env = DummyVecEnv([lambda: GDEnv()])
    env = VecFrameStack(env, n_stack=4)
    return env


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs("modelos_guardados", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("== EWC sobre QR-DQN: cubo 1 -> cubo 2 ==")
    print(f"   Checkpoint  : {SOURCE_CHECKPOINT}")
    print(f"   Fisher path : {FISHER_PATH}")
    print(f"   Pasos       : {TOTAL_TIMESTEPS:,}")
    print(f"   Buffer      : {EWC_BUFFER_SIZE:,}")
    print(f"   Lambda      : {EWC_LAMBDA}")
    print()

    env = build_env()

    print(f"Cargando modelo {SOURCE_CHECKPOINT}...")
    model = EWCQRDQN.load(
        SOURCE_CHECKPOINT,
        env=env,
        device="auto",
        custom_objects={
            "learning_rate":           EWC_LEARNING_RATE,
            "exploration_initial_eps": EWC_EXPLORATION_INITIAL,
            "exploration_final_eps":   EWC_EXPLORATION_FINAL,
            "exploration_fraction":    EWC_EXPLORATION_FRACTION,
            "learning_starts":         EWC_LEARNING_STARTS,
            "buffer_size":             EWC_BUFFER_SIZE,
            "target_update_interval":  EWC_TARGET_UPDATE,
        },
    )
    model.ewc_lambda = EWC_LAMBDA  # asegurar valor tras load

    # ---------- Fase 1: Fisher (solo si no existe el .pt) -----------
    if os.path.isfile(FISHER_PATH):
        print(f"Fisher previa detectada en {FISHER_PATH}. Saltando fase 1.")
        model.load_fisher(FISHER_PATH)
    else:
        print()
        print("FASE 1 - Calculo de la Fisher.")
        print("Pon GD en CUBO 1 (Start Position al inicio del nivel).")
        print("Empieza la captura en 8 segundos...")
        time.sleep(8)

        model.compute_fisher(env, n_samples=EWC_FISHER_SAMPLES)
        model.save_fisher(FISHER_PATH)
        print()
        print("FASE 1 completada. AHORA cambia GD a CUBO 2 (Start Position 2a parte).")
        print("La fase 2 arrancara en 15 segundos para que tengas tiempo.")
        time.sleep(15)

    # ---------- Fase 2: entrenamiento con EWC en cubo 2 ------------
    print()
    print("FASE 2 - Entrenamiento con EWC en cubo 2.")
    print("Comprueba que GD esta en cubo 2 y NO toques.")
    print("Empieza el entrenamiento en 5 segundos...")
    time.sleep(5)

    callback = MetricsAndSaveCallback(
        save_freq=SAVE_FREQ,
        save_path=f"modelos_guardados/gd_{RUN_NAME}",
        run_name=RUN_NAME,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        reset_num_timesteps=True,
        log_interval=10,
    )

    final_path = f"modelos_guardados/gd_{RUN_NAME}_FINAL.zip"
    model.save(final_path)
    save_plots(callback.history, RUN_NAME, show=False)
    env.close()

    print()
    print(f"Modelo final  : {final_path}")
    print(f"Fisher        : {FISHER_PATH}")
    print(f"Metricas      : {METRICS_DIR}/{RUN_NAME}_metrics.csv")
    print(f"Grafica       : {METRICS_DIR}/{RUN_NAME}_plot.png")


if __name__ == "__main__":
    main()
