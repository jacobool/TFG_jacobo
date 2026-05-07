"""
gd_rl_cubo2_ewc.py
Entrena la segunda sección de cubo usando EWC (Elastic Weight Consolidation)
para evitar que el agente olvide la primera sección.

Flujo de ejecución:
  FASE 1 — Juego en el nivel CUBO 1:
    Carga el modelo entrenado y calcula la Fisher Information Matrix.
  FASE 2 — Usuario reposiciona el juego al nivel CUBO 2:
    Entrena con la penalización EWC que ancla los pesos importantes al cubo 1.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pygetwindow as gw
import ctypes
from ctypes import wintypes
import time
import os
import pydirectinput
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

# ← CAMBIAR a la ruta del modelo final del cubo 1
MODELO_CUBO1 = "models/gd_dqn_FINAL_4.zip"

GAME_TITLE = "Geometry Dash"
PLAYER_X_REL = 0.345
PLAYER_BAND_W = 0.065
LOWER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
GREEN_AREA_MIN = 600

DEATH_FRAMES_NEEDED = 2
STEP_DURATION = 1 / 15
MIN_EPISODE_GAP = 1.1

METRICS_PATH = "cubo2_ewc_metrics.csv"
SAVE_FREQ = 10000
TOTAL_STEPS = 300000

# EWC — cuánto se penaliza cambiar pesos importantes del cubo 1.
# Rango típico: 1000–20000. Más alto = menos olvido pero aprendizaje más lento.
EWC_LAMBDA = 5000
# Pasos del juego (en el cubo 1) para estimar la Fisher. 300–1000 es suficiente.
EWC_FISHER_SAMPLES = 500


# ─── FUNCIONES AUXILIARES ────────────────────────────────────────────────────

def get_window_rect(hwnd):
    rect = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    point = wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
    return {"top": point.y, "left": point.x, "width": w, "height": h}


def detect_player_and_band(frame_bgr):
    h, w = frame_bgr.shape[:2]
    player_x = int(w * PLAYER_X_REL)
    band_half = int(w * PLAYER_BAND_W / 2)
    x1 = max(0, player_x - band_half)
    x2 = min(w, player_x + band_half)
    y1, y2 = int(h * 0.08), int(h * 0.92)
    band_bgr = frame_bgr[y1:y2, x1:x2]

    player_center_y = h / 2
    green_area = 0
    bbox_abs = None

    if band_bgr.size > 0:
        band_hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(band_hsv, LOWER_GREEN, UPPER_GREEN)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        green_area = cv2.countNonZero(mask_clean)
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = max(contours, key=cv2.contourArea) if contours else None

        if best_cnt is not None and cv2.contourArea(best_cnt) > 50:
            bx, by, bw, bh = cv2.boundingRect(best_cnt)
            abs_cy = y1 + by + bh // 2
            player_center_y = abs_cy
            bbox_abs = (x1 + bx, y1 + by, bw, bh)

    return player_center_y, green_area, bbox_abs


def save_plots(history, show=False):
    if not history:
        return
    df = pd.DataFrame(history)
    window = min(30, len(df))

    fig, axs = plt.subplots(4, 1, figsize=(12, 18))

    axs[0].plot(df['episode'], df['time_alive'], color='blue', alpha=0.25, linewidth=0.8)
    df['time_smooth'] = df['time_alive'].rolling(window=window, min_periods=1).mean()
    axs[0].plot(df['episode'], df['time_smooth'], color='darkblue', linewidth=2.5,
                label=f'Media {window} ep')
    axs[0].axhline(y=df['time_alive'].max(), color='gold', linestyle='--', alpha=0.7,
                   label=f'Récord: {df["time_alive"].max():.1f}s')
    axs[0].set_title('Tiempo de Supervivencia (Cubo 2 — EWC)')
    axs[0].set_ylabel('Segundos')
    axs[0].legend()

    axs[1].plot(df['episode'], df['reward'], color='green', alpha=0.25, linewidth=0.8)
    df['reward_smooth'] = df['reward'].rolling(window=window, min_periods=1).mean()
    axs[1].plot(df['episode'], df['reward_smooth'], color='darkgreen', linewidth=2.5)
    axs[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axs[1].set_title('Recompensa Total')
    axs[1].set_ylabel('Reward')

    axs[2].plot(df['episode'], df['loss'], color='red', alpha=0.25, linewidth=0.8)
    df['loss_smooth'] = df['loss'].rolling(window=window, min_periods=1).mean()
    axs[2].plot(df['episode'], df['loss_smooth'], color='darkred', linewidth=2.5)
    axs[2].set_title('Pérdida DQN')
    axs[2].set_ylabel('Huber Loss')

    # Loss EWC: cuánto presiona la regularización (debería estabilizarse)
    axs[3].plot(df['episode'], df['ewc_loss'], color='purple', alpha=0.25, linewidth=0.8)
    df['ewc_smooth'] = df['ewc_loss'].rolling(window=window, min_periods=1).mean()
    axs[3].plot(df['episode'], df['ewc_smooth'], color='darkviolet', linewidth=2.5)
    axs[3].set_title('Penalización EWC (debe estabilizarse si no hay olvido)')
    axs[3].set_ylabel('EWC Penalty')

    plt.tight_layout()
    plt.savefig('cubo2_ewc_plot.png', dpi=120)
    df.to_csv(METRICS_PATH, index=False)

    if show:
        plt.show()
    plt.close(fig)


# ─── ENTORNO ─────────────────────────────────────────────────────────────────

class GDEnvCubo(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)
        self.sct = mss.mss()
        windows = gw.getWindowsWithTitle(GAME_TITLE)
        if not windows:
            raise ValueError("No se encontró la ventana de Geometry Dash")
        self.win = windows[0]
        self.hwnd = self.win._hWnd
        self.win.activate()
        self.attempt = 1
        self.last_action_time = time.perf_counter()
        self._no_green_count = 0
        self._ep_frames = 0
        self._best_frames = 0
        self.last_episode_time = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._no_green_count = 0
        self._ep_frames = 0

        while True:
            frame_bgr, frame_gray = self._capture_frame()
            _, green_area, bbox_abs = detect_player_and_band(frame_bgr)
            ahora = time.perf_counter()
            if green_area >= GREEN_AREA_MIN and (ahora - self.last_episode_time) >= MIN_EPISODE_GAP:
                self.last_episode_time = ahora
                self.attempt += 1
                print(f"[C2-EWC] Episodio {self.attempt} — green: {green_area}")
                break
            time.sleep(STEP_DURATION)

        return self._get_obs(frame_gray, frame_bgr.shape[1], bbox_abs), {}

    def step(self, action):
        step_start = time.perf_counter()
        reward = 0.05
        done = False

        now = time.perf_counter()
        if action == 1 and (now - self.last_action_time > 0.05):
            pydirectinput.keyDown("space")
            pydirectinput.keyUp("space")
            reward -= 0.04
            self.last_action_time = now

        frame_bgr, frame_gray = self._capture_frame()
        _, green_area, bbox_abs = detect_player_and_band(frame_bgr)
        player_visible = green_area >= GREEN_AREA_MIN

        if player_visible:
            self._no_green_count = 0
            self._ep_frames += 1
            if self._ep_frames > self._best_frames:
                reward += 0.5
        else:
            self._no_green_count += 1
            if self._no_green_count >= DEATH_FRAMES_NEEDED:
                reward = -5.0
                done = True
                self._no_green_count = 0
                if self._ep_frames > self._best_frames:
                    self._best_frames = self._ep_frames
            else:
                reward = 0.0

        obs = self._get_obs(frame_gray, frame_bgr.shape[1], bbox_abs if player_visible else None)

        elapsed = time.perf_counter() - step_start
        if STEP_DURATION - elapsed > 0:
            time.sleep(STEP_DURATION - elapsed)

        return obs, reward, done, False, {"attempt": self.attempt, "green": green_area}

    def _capture_frame(self):
        monitor = get_window_rect(self.hwnd)
        img = np.array(self.sct.grab(monitor))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = frame_bgr.shape[:2]
        x1_ai, x2_ai = int(w * 0.20), int(w * 0.90)
        frame_gray_cropped = cv2.cvtColor(img[:, x1_ai:x2_ai], cv2.COLOR_BGRA2GRAY)
        return frame_bgr, frame_gray_cropped

    def _get_obs(self, frame_gray_cropped, original_width, bbox_abs):
        _, thresh = cv2.threshold(frame_gray_cropped, 205, 255, cv2.THRESH_BINARY)
        kernel_ai = np.ones((2, 2), np.uint8)
        vision_clean = cv2.dilate(thresh, kernel_ai, iterations=1)

        if bbox_abs is not None:
            x_abs, y_abs, bw, bh = bbox_abs
            x1_ai = int(original_width * 0.20)
            p_x1 = x_abs - x1_ai
            if p_x1 >= 0:
                cv2.rectangle(vision_clean, (p_x1, y_abs), (p_x1 + bw, y_abs + bh), 255, -1)

        resized = cv2.resize(vision_clean, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1).astype(np.uint8)

    def close(self):
        self.sct.close()


# ─── DQN CON EWC ─────────────────────────────────────────────────────────────

class EWCDQN(DQN):
    """
    DQN con regularización EWC.
    Añade al loss de Bellman una penalización proporcional a cuánto se alejan
    los pesos actuales de los óptimos del cubo 1, ponderada por la importancia
    de cada peso (Fisher Information Matrix diagonal).
    """

    def __init__(self, *args, ewc_lambda: float = EWC_LAMBDA, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.ewc_fisher: dict | None = None      # F_i (importancia de cada peso)
        self.ewc_star_params: dict | None = None  # θ*_i (pesos óptimos cubo 1)
        self._last_ewc_loss = 0.0                # para logging en el callback

    # ── Cálculo de la Fisher Information Matrix ───────────────────────────────

    def compute_fisher(self, vec_env, n_samples: int = EWC_FISHER_SAMPLES) -> None:
        """
        Estima la FIM diagonal usando el gradiente del log de la política greedy
        evaluada sobre n_samples pasos del entorno del cubo 1.

        Requiere que vec_env esté capturando frames del cubo 1 en ese momento.
        """
        print(f"\n[EWC] Calculando Fisher sobre {n_samples} pasos del cubo 1...")
        self.policy.set_training_mode(False)

        fisher: dict[str, th.Tensor] = {
            n: th.zeros_like(p)
            for n, p in self.policy.q_net.named_parameters()
            if p.requires_grad
        }

        obs = vec_env.reset()
        count = 0
        while count < n_samples:
            obs_th, _ = self.policy.obs_to_tensor(obs)
            q_values = self.policy.q_net(obs_th)

            # Usamos log_softmax sobre los Q-values como proxy de log π(a|s)
            log_probs = F.log_softmax(q_values, dim=-1)
            action = q_values.argmax(dim=-1)
            selected_log_prob = log_probs[th.arange(len(action)), action].sum()

            self.policy.optimizer.zero_grad()
            selected_log_prob.backward()

            for n, p in self.policy.q_net.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)

            count += 1
            obs, _, dones, _ = vec_env.step(action.cpu().numpy())
            if dones.any():
                obs = vec_env.reset()

        for n in fisher:
            fisher[n] /= n_samples

        # Guardamos tanto la Fisher como los pesos θ* en el device correcto
        self.ewc_fisher = fisher
        self.ewc_star_params = {
            n: p.detach().clone()
            for n, p in self.policy.q_net.named_parameters()
            if p.requires_grad
        }

        self.policy.set_training_mode(True)
        total_params = sum(f.numel() for f in fisher.values())
        mean_fisher = sum(f.mean().item() for f in fisher.values()) / len(fisher)
        print(f"[EWC] Fisher calculada. Parámetros: {total_params:,} | F media: {mean_fisher:.2e}")
        print(f"[EWC] λ={self.ewc_lambda}  →  penalización inicial estimada: "
              f"≈ {self.ewc_lambda * mean_fisher:.2e} por parámetro\n")

    # ── Override del bucle de entrenamiento ───────────────────────────────────

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        dqn_losses = []
        ewc_penalties = []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                next_q = self.q_net_target(replay_data.next_observations)
                next_q, _ = next_q.max(dim=1)
                next_q = next_q.reshape(-1, 1)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q

            current_q = self.q_net(replay_data.observations)
            current_q = th.gather(current_q, dim=1, index=replay_data.actions.long())

            dqn_loss = F.smooth_l1_loss(current_q, target_q)

            # ── Penalización EWC ──────────────────────────────────────────────
            ewc_penalty = th.tensor(0.0, device=self.device)
            if self.ewc_fisher is not None:
                for n, p in self.policy.q_net.named_parameters():
                    if n in self.ewc_fisher and n in self.ewc_star_params:
                        diff = p - self.ewc_star_params[n]
                        ewc_penalty = ewc_penalty + (self.ewc_fisher[n] * diff.pow(2)).sum()

            loss = dqn_loss + (self.ewc_lambda / 2) * ewc_penalty

            dqn_losses.append(dqn_loss.item())
            ewc_penalties.append(ewc_penalty.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self._last_ewc_loss = float(np.mean(ewc_penalties))

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(dqn_losses)))
        self.logger.record("train/ewc_penalty", self._last_ewc_loss)


# ─── CALLBACK ────────────────────────────────────────────────────────────────

class MetricsAndSaveCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.history = []
        self.episode_count = 0
        self.ep_reward = 0.0
        self.ep_start_time = time.time()

    def _on_step(self) -> bool:
        self.ep_reward += self.locals["rewards"][0]

        if self.locals["dones"][0]:
            self.episode_count += 1
            time_alive = time.time() - self.ep_start_time

            loss = 0.0
            if hasattr(self.model, "logger") and "train/loss" in self.model.logger.name_to_value:
                loss = self.model.logger.name_to_value["train/loss"]

            ewc_loss = getattr(self.model, "_last_ewc_loss", 0.0)

            self.history.append({
                'episode': self.episode_count,
                'time_alive': time_alive,
                'reward': self.ep_reward,
                'loss': loss,
                'ewc_loss': ewc_loss,
            })

            self.ep_reward = 0.0
            self.ep_start_time = time.time()

        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            model_name = f"{self.save_path}_{self.num_timesteps}_steps"
            self.model.save(model_name)
            save_plots(self.history, show=False)
            print(f"[SAVE] Paso {self.num_timesteps} — modelo guardado: {model_name}.zip")

        return True


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("modelos_guardados", exist_ok=True)

    # Creamos el entorno (mismo código para cubo 1 y cubo 2 — solo cambia el nivel en el juego)
    env = DummyVecEnv([lambda: GDEnvCubo()])
    env = VecFrameStack(env, n_stack=4)

    # ── FASE 1: cargar modelo cubo 1 y calcular Fisher ────────────────────────
    print("=" * 60)
    print("  FASE 1 — CÁLCULO DE FISHER (cubo 1)")
    print("=" * 60)
    print(f"  Cargando modelo: {MODELO_CUBO1}")
    print("  Asegúrate de que el juego está EN EL NIVEL DEL CUBO 1")
    print("  y que el jugador puede moverse normalmente.")
    input("\n  Pulsa ENTER para empezar la recogida de datos de Fisher...\n")

    # Cargamos el modelo del cubo 1 como DQN estándar para extraer sus pesos
    cube1_model = DQN.load(MODELO_CUBO1, env=env, device="auto")

    # Creamos el modelo EWC con los mismos hiperparámetros que el cubo 1
    ewc_model = EWCDQN(
        "CnnPolicy", env,
        ewc_lambda=EWC_LAMBDA,
        verbose=1,
        buffer_size=50000,
        learning_starts=5000,
        batch_size=64,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2000,
        learning_rate=2e-5,       # LR más bajo que en cubo1: ya tenemos buena base
        gamma=0.99,
        exploration_fraction=0.15, # Menos exploración: ya sabe cómo jugar al cubo
        exploration_final_eps=0.02,
        device="auto",
    )

    # Copiamos los pesos del cubo 1 al modelo EWC
    ewc_model.set_parameters(cube1_model.get_parameters())
    del cube1_model  # Liberar memoria

    # Calculamos la Fisher con el entorno del cubo 1
    ewc_model.compute_fisher(env, n_samples=EWC_FISHER_SAMPLES)

    # ── FASE 2: entrenar en el cubo 2 ─────────────────────────────────────────
    print("=" * 60)
    print("  FASE 2 — ENTRENAMIENTO EN CUBO 2 (con EWC activo)")
    print("=" * 60)
    print("  Ahora reposiciona el juego en el nivel del CUBO 2.")
    print(f"  Se entrenarán {TOTAL_STEPS:,} pasos.")
    print(f"  EWC λ = {EWC_LAMBDA}  |  LR = 2e-5  |  Guardado cada {SAVE_FREQ:,} pasos.")
    input("\n  Pulsa ENTER cuando el juego esté en el CUBO 2...\n")

    callback = MetricsAndSaveCallback(
        save_freq=SAVE_FREQ,
        save_path="modelos_guardados/cubo2_ewc"
    )

    ewc_model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

    print("\n[FIN] Entrenamiento completado.")
    ewc_model.save("modelos_guardados/cubo2_ewc_FINAL")
    print("Modelo guardado: modelos_guardados/cubo2_ewc_FINAL.zip")
    save_plots(callback.history, show=True)
    env.close()
