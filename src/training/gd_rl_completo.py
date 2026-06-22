import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

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
from stable_baselines3.common.callbacks import BaseCallback

# --- 1. CONFIGURACIÓN ---
GAME_TITLE = "Geometry Dash"
PLAYER_X_REL = 0.345
PLAYER_BAND_W = 0.08
LOWER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([45, 255, 255], dtype=np.uint8)

# Rangos de área para distinguir modo
CUBE_AREA_MIN  = 600           # blob grande  → cubo
SHIP_AREA_MIN  = 130           # blob pequeño → nave
SHIP_AREA_MAX  = 300

# Hitbox nave
SHIP_HITBOX_WIDTH  = 55
SHIP_HITBOX_HEIGHT = 45
SHIP_OFFSET_X      = 0
SHIP_OFFSET_Y      = 10

# Entrenamiento
DEATH_FRAMES_NEEDED  = 3
STEP_DURATION        = 1 / 15
MIN_EPISODE_GAP      = 1.6
METRICS_PATH         = "completo_metrics.csv"
SAVE_FREQ            = 40000
MIN_ACTION_HOLD_STEPS = 3
SWITCH_PENALTY        = 0.015
EDGE_PENALTY          = 0.05

# Checkpoint de inicio (pon None para entrenar desde cero)
# Partimos del modelo de nave porque fue la parte más costosa de entrenar.
# El cubo se reaprendrá rápido al ser la primera sección de cada episodio.
CHECKPOINT_RESUME = "modelos_guardados/nave_dqn_FINAL_3"


# --- 2. DETECCIÓN UNIFICADA ---
def detect_player_unified(frame_bgr):
    """
    Devuelve (mode, player_center_y, green_area, bbox_abs).
    mode: 'cube' | 'ship' | None
    """
    h, w = frame_bgr.shape[:2]
    player_x  = int(w * PLAYER_X_REL)
    band_half = int(w * PLAYER_BAND_W / 2)
    x1 = max(0, player_x - band_half)
    x2 = min(w, player_x + band_half)
    y1, y2 = int(h * 0.08), int(h * 0.92)
    band_bgr = frame_bgr[y1:y2, x1:x2]

    if band_bgr.size == 0:
        return None, h / 2, 0, None

    band_hsv  = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
    mask      = cv2.inRange(band_hsv, LOWER_GREEN, UPPER_GREEN)
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask      = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask      = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, h / 2, 0, None

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    bx, by, bw, bh = cv2.boundingRect(best)

    # --- Nave ---
    if SHIP_AREA_MIN <= area <= SHIP_AREA_MAX:
        cx = bx + bw // 2
        cy = by + bh // 2
        ex = max(0, cx - SHIP_HITBOX_WIDTH  // 2 + SHIP_OFFSET_X)
        ey = max(0, cy - SHIP_HITBOX_HEIGHT // 2 + SHIP_OFFSET_Y)
        abs_cy  = y1 + cy + SHIP_OFFSET_Y
        bbox    = (x1 + ex, y1 + ey, SHIP_HITBOX_WIDTH, SHIP_HITBOX_HEIGHT)
        return 'ship', abs_cy, area, bbox

    # --- Cubo ---
    if area >= CUBE_AREA_MIN:
        abs_cy = y1 + by + bh // 2
        bbox   = (x1 + bx, y1 + by, bw, bh)
        return 'cube', abs_cy, area, bbox

    return None, h / 2, 0, None


# --- 3. FUNCIONES AUXILIARES ---
def get_window_rect(hwnd):
    rect  = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h  = rect.right - rect.left, rect.bottom - rect.top
    point = wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
    return {"top": point.y, "left": point.x, "width": w, "height": h}


def save_plots(history, show=False):
    if not history:
        return
    df     = pd.DataFrame(history)
    window = min(30, len(df))
    fig, axs = plt.subplots(3, 1, figsize=(12, 14))

    for ax, col, color, title, ylabel in [
        (axs[0], 'time_alive', 'blue',  'Tiempo de Supervivencia (COMPLETO)', 'Segundos'),
        (axs[1], 'reward',     'green', 'Recompensa Total',                    'Reward'),
        (axs[2], 'loss',       'red',   'Pérdida (Loss) Promedio',             'MSE Loss'),
    ]:
        ax.plot(df['episode'], df[col], color=color, alpha=0.25, linewidth=0.8)
        smooth = df[col].rolling(window=window, min_periods=1).mean()
        ax.plot(df['episode'], smooth, color=f'dark{color}', linewidth=2.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig('completo_metrics_plot.png', dpi=120)
    df.to_csv(METRICS_PATH, index=False)
    if show:
        plt.show()
    plt.close(fig)


# --- 4. ENTORNO ---
class GDEnvCompleto(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space      = spaces.Discrete(2)

        self.sct     = mss.mss()
        windows      = gw.getWindowsWithTitle(GAME_TITLE)
        if not windows:
            raise ValueError("No se encontró la ventana de Geometry Dash")
        self.win  = windows[0]
        self.hwnd = self.win._hWnd
        self.win.activate()

        self.attempt           = 1
        self._no_green_count   = 0
        self.last_episode_time = 0.0
        self.is_pressing       = False
        self.control_state     = 0
        self.control_lock_steps = 0
        self._first_no_green_obs = None

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._no_green_count    = 0
        self._first_no_green_obs = None

        if self.is_pressing:
            pydirectinput.keyUp("space")
            self.is_pressing = False
        self.control_state      = 0
        self.control_lock_steps = 0

        # Espera a que el jugador (cubo) aparezca al inicio del nivel
        while True:
            frame_bgr, frame_crop = self._capture_frame()
            mode, _, area, bbox   = detect_player_unified(frame_bgr)
            ahora = time.perf_counter()
            player_ok = (mode == 'cube') and (area >= CUBE_AREA_MIN)
            if player_ok and (ahora - self.last_episode_time) >= MIN_EPISODE_GAP:
                self.last_episode_time = ahora
                self.attempt += 1
                print(f"Episodio {self.attempt} | modo={mode} | area={area:.0f}")
                break
            time.sleep(STEP_DURATION)

        return self._get_obs(frame_crop, frame_bgr.shape[1], bbox), {}

    # ------------------------------------------------------------------
    def step(self, action):
        step_start = time.perf_counter()
        reward     = 0.08
        done       = False

        # Control con bloqueo anti-oscilación
        desired = int(action)
        if self.control_lock_steps > 0:
            desired = self.control_state
            self.control_lock_steps -= 1
        else:
            if desired != self.control_state:
                reward -= SWITCH_PENALTY
                self.control_state      = desired
                self.control_lock_steps = MIN_ACTION_HOLD_STEPS - 1

        if self.control_state == 1:
            if not self.is_pressing:
                pydirectinput.keyDown("space")
                self.is_pressing = True
        else:
            if self.is_pressing:
                pydirectinput.keyUp("space")
                self.is_pressing = False

        frame_bgr, frame_crop                 = self._capture_frame()
        mode, player_cy, area, bbox_abs = detect_player_unified(frame_bgr)
        player_visible = mode is not None

        # Edge penalty
        h      = frame_bgr.shape[0]
        y_norm = np.clip(player_cy / max(1, h), 0.0, 1.0)
        if y_norm < 0.27 or y_norm > 0.73:
            reward -= EDGE_PENALTY

        # Lógica de muerte con buffer retroactivo
        if player_visible:
            self._no_green_count    = 0
            self._first_no_green_obs = None
            obs = self._get_obs(frame_crop, frame_bgr.shape[1], bbox_abs)
        else:
            self._no_green_count += 1
            if self._no_green_count == 1:
                self._first_no_green_obs = self._get_obs(frame_crop, frame_bgr.shape[1], None)
                reward = 0.0
                obs    = self._first_no_green_obs
            elif self._no_green_count < DEATH_FRAMES_NEEDED:
                reward = 0.0
                obs    = self._first_no_green_obs
            else:
                reward               = -5.0
                done                 = True
                self._no_green_count = 0
                obs                  = self._first_no_green_obs
                self._first_no_green_obs = None

        elapsed = time.perf_counter() - step_start
        if STEP_DURATION - elapsed > 0:
            time.sleep(STEP_DURATION - elapsed)

        return obs, reward, done, False, {
            "attempt": self.attempt,
            "mode":    mode,
            "area":    area,
        }

    # ------------------------------------------------------------------
    def _capture_frame(self):
        monitor        = get_window_rect(self.hwnd)
        img            = np.array(self.sct.grab(monitor))
        frame_bgr      = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w           = frame_bgr.shape[:2]
        x1             = int(w * 0.20)
        x2             = int(w * 0.90)
        frame_crop     = frame_bgr[:, x1:x2]
        return frame_bgr, frame_crop

    def _get_obs(self, frame_crop, original_width, bbox_abs):
        lower_white  = np.array([220, 220, 220], dtype=np.uint8)
        upper_white  = np.array([255, 255, 255], dtype=np.uint8)
        mask         = cv2.inRange(frame_crop, lower_white, upper_white)
        kernel       = np.ones((2, 2), np.uint8)
        vision       = cv2.dilate(mask, kernel, iterations=1)

        if bbox_abs is not None:
            x_abs, y_abs, bw, bh = bbox_abs
            x1_ai = int(original_width * 0.20)
            px    = x_abs - x1_ai
            if px >= 0:
                cv2.rectangle(vision, (px, y_abs), (px + bw, y_abs + bh), 255, -1)

        resized = cv2.resize(vision, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1).astype(np.uint8)

    def close(self):
        if self.is_pressing:
            pydirectinput.keyUp("space")
        self.sct.close()


# --- 5. CALLBACK ---
class MetricsAndSaveCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq     = save_freq
        self.save_path     = save_path
        self.history       = []
        self.episode_count = 0
        self.ep_reward     = 0.0
        self.ep_start_time = time.time()

    def _on_step(self) -> bool:
        self.ep_reward += self.locals["rewards"][0]

        if self.locals["dones"][0]:
            self.episode_count += 1
            time_alive = time.time() - self.ep_start_time
            loss = 0.0
            if hasattr(self.model, "logger") and "train/loss" in self.model.logger.name_to_value:
                loss = self.model.logger.name_to_value["train/loss"]
            self.history.append({
                'episode':    self.episode_count,
                'time_alive': time_alive,
                'reward':     self.ep_reward,
                'loss':       loss,
            })
            self.ep_reward     = 0.0
            self.ep_start_time = time.time()

        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            name = f"{self.save_path}_{self.num_timesteps}_steps"
            self.model.save(name)
            save_plots(self.history, show=False)
            print(f"[AUTO-SAVE] Paso {self.num_timesteps} → {name}.zip")

        return True


# --- 6. ENTRENAMIENTO ---
if __name__ == "__main__":
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    env = DummyVecEnv([lambda: GDEnvCompleto()])
    env = VecFrameStack(env, n_stack=4)

    os.makedirs("modelos_guardados", exist_ok=True)
    callback = MetricsAndSaveCallback(
        save_freq=SAVE_FREQ,
        save_path="modelos_guardados/completo_dqn"
    )

    if CHECKPOINT_RESUME and os.path.exists(CHECKPOINT_RESUME + ".zip"):
        print(f"Cargando checkpoint: {CHECKPOINT_RESUME}")
        model = DQN.load(
            CHECKPOINT_RESUME,
            env=env,
            custom_objects={
                "learning_rate":           5e-5,
                "exploration_initial_eps": 0.15,
                "exploration_final_eps":   0.02,
                "exploration_fraction":    0.3,
                "buffer_size":             100000,
            }
        )
        model.learning_starts = 0
    else:
        print("Entrenando desde cero.")
        model = DQN(
            "CnnPolicy", env,
            verbose=1,
            buffer_size=100000,
            learning_starts=5000,
            batch_size=64,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=2000,
            learning_rate=7.5e-5,
            gamma=0.995,
            exploration_fraction=0.35,
            exploration_final_eps=0.02,
        )

    print("Iniciando entrenamiento completo (Cubo 1 → Nave → Cubo 2).")
    print(f"Guardado automático cada {SAVE_FREQ} pasos.")
    print("Pon el juego en primer plano con el nivel completo abierto.\n")

    model.learn(total_timesteps=900000, callback=callback)

    model.save("modelos_guardados/completo_dqn_FINAL")
    print("Modelo guardado: completo_dqn_FINAL.zip")
    save_plots(callback.history, show=True)
    env.close()
