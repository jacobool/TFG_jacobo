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
GREEN_AREA_MIN = 130
GREEN_AREA_MAX = 300

SHIP_HITBOX_WIDTH = 55
SHIP_HITBOX_HEIGHT = 45
SHIP_OFFSET_X = 0
SHIP_OFFSET_Y = 10

# Constantes de Entrenamiento
DEATH_FRAMES_NEEDED = 3
STEP_DURATION = 1 / 15
MIN_EPISODE_GAP = 1.6
METRICS_PATH = "nave_metrics_FINAL_2.csv"
SAVE_FREQ = 40000
MIN_ACTION_HOLD_STEPS = 5
SWITCH_PENALTY = 0.015
EDGE_PENALTY = 0.05

# Checkpoint desde el que se reanuda
CHECKPOINT_RESUME = "modelos_guardados/nave_dqn_2_960000_steps"


# --- 2. FUNCIONES AUXILIARES Y GRÁFICAS ---
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

        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = max(contours, key=cv2.contourArea) if contours else None

        if best_cnt is not None:
            largest_piece_area = cv2.contourArea(best_cnt)
            if largest_piece_area >= GREEN_AREA_MIN and largest_piece_area <= GREEN_AREA_MAX:
                bx, by, bw, bh = cv2.boundingRect(best_cnt)
                center_x_local = bx + bw // 2
                center_y_local = by + bh // 2

                expanded_x = center_x_local - (SHIP_HITBOX_WIDTH // 2) + SHIP_OFFSET_X
                expanded_y = center_y_local - (SHIP_HITBOX_HEIGHT // 2) + SHIP_OFFSET_Y
                expanded_x = max(0, expanded_x)
                expanded_y = max(0, expanded_y)

                abs_cy = y1 + center_y_local + SHIP_OFFSET_Y
                player_center_y = abs_cy

                bbox_abs = (x1 + expanded_x, y1 + expanded_y, SHIP_HITBOX_WIDTH, SHIP_HITBOX_HEIGHT)
                green_area = largest_piece_area
            else:
                green_area = 0
                bbox_abs = None

    return player_center_y, green_area, bbox_abs


def save_plots(history, show=False):
    if not history:
        return
    df = pd.DataFrame(history)
    window = min(30, len(df))

    fig, axs = plt.subplots(3, 1, figsize=(12, 14))

    axs[0].plot(df['episode'], df['time_alive'], color='blue', alpha=0.25, linewidth=0.8)
    df['time_smooth'] = df['time_alive'].rolling(window=window, min_periods=1).mean()
    axs[0].plot(df['episode'], df['time_smooth'], color='darkblue', linewidth=2.5, label=f'Media {window} ep')
    axs[0].axhline(y=df['time_alive'].max(), color='gold', linestyle='--', alpha=0.7)
    axs[0].set_title('Tiempo de Supervivencia (NAVE)')
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
    axs[2].set_title('Pérdida (Loss) Promedio')
    axs[2].set_ylabel('MSE Loss')

    plt.tight_layout()
    plt.savefig('nave_metrics_plot_FINAL_2.png', dpi=120)
    df.to_csv(METRICS_PATH, index=False)

    if show:
        plt.show()
    plt.close(fig)


# --- 3. CLASE DEL ENTORNO ---
class GDEnv(gym.Env):
    def __init__(self):
        super(GDEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)
        self.sct = mss.mss()
        self.windows = gw.getWindowsWithTitle(GAME_TITLE)
        if not self.windows:
            raise ValueError("No Geometry Dash window found")
        self.win = self.windows[0]
        self.hwnd = self.win._hWnd
        self.win.activate()

        self.attempt = 1
        self._no_green_count = 0
        self.last_episode_time = 0.0
        self.is_pressing = False
        self.control_state = 0
        self.control_lock_steps = 0

        # Buffer retroactivo: guarda la obs del primer frame sin verde
        # (colisión real) para que la penalización de muerte se asocie
        # al estado visual correcto y no al frame de reaparición.
        self._first_no_green_obs = None
        self._first_no_green_reward = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._no_green_count = 0
        self._first_no_green_obs = None
        self._first_no_green_reward = None

        if self.is_pressing:
            pydirectinput.keyUp("space")
            self.is_pressing = False
        self.control_state = 0
        self.control_lock_steps = 0

        while True:
            frame_bgr, frame_color_cropped = self._capture_frame()
            _, green_area, bbox_abs = detect_player_and_band(frame_bgr)

            ahora = time.perf_counter()
            if green_area >= GREEN_AREA_MIN and green_area <= GREEN_AREA_MAX and (ahora - self.last_episode_time) >= MIN_EPISODE_GAP:
                self.last_episode_time = ahora
                self.attempt += 1
                print(f"🚀 NAVE | Episodio {self.attempt} iniciado. Green area: {green_area}")
                break
            time.sleep(STEP_DURATION)

        return self._get_obs(frame_color_cropped, frame_bgr.shape[1], bbox_abs), {}

    def step(self, action):
        step_start = time.perf_counter()
        reward = 0.08
        done = False

        # --- Control con bloqueo anti-oscilación ---
        desired_action = int(action)
        if self.control_lock_steps > 0:
            desired_action = self.control_state
            self.control_lock_steps -= 1
        else:
            if desired_action != self.control_state:
                reward -= SWITCH_PENALTY
                self.control_state = desired_action
                self.control_lock_steps = MIN_ACTION_HOLD_STEPS - 1

        if self.control_state == 1:
            if not self.is_pressing:
                pydirectinput.keyDown("space")
                self.is_pressing = True
        else:
            if self.is_pressing:
                pydirectinput.keyUp("space")
                self.is_pressing = False

        frame_bgr, frame_color_cropped = self._capture_frame()
        player_center_y, green_area, bbox_abs = detect_player_and_band(frame_bgr)

        # --- Edge penalty (umbrales ampliados: 0.27/0.73) ---
        h = frame_bgr.shape[0]
        y_norm = np.clip(player_center_y / max(1, h), 0.0, 1.0)
        if y_norm < 0.27 or y_norm > 0.73:
            reward -= EDGE_PENALTY

        # --- Lógica de muerte con buffer retroactivo ---
        if green_area >= GREEN_AREA_MIN and green_area <= GREEN_AREA_MAX:
            self._no_green_count = 0
            self._first_no_green_obs = None
            self._first_no_green_reward = None
            obs = self._get_obs(frame_color_cropped, frame_bgr.shape[1], bbox_abs)

        else:
            self._no_green_count += 1

            if self._no_green_count == 1:
                # Primer frame sin verde: guardamos el estado de la colisión real
                self._first_no_green_obs = self._get_obs(
                    frame_color_cropped, frame_bgr.shape[1], bbox_abs
                )
                self._first_no_green_reward = 0.0
                reward = 0.0
                obs = self._first_no_green_obs

            elif self._no_green_count < DEATH_FRAMES_NEEDED:
                # Frames de tolerancia: mantenemos el frame de colisión
                reward = 0.0
                obs = self._first_no_green_obs

            else:
                # Muerte confirmada: devolvemos el frame de colisión real
                reward = -5.0
                done = True
                self._no_green_count = 0
                obs = self._first_no_green_obs
                self._first_no_green_obs = None
                self._first_no_green_reward = None

        elapsed = time.perf_counter() - step_start
        if STEP_DURATION - elapsed > 0:
            time.sleep(STEP_DURATION - elapsed)

        return obs, reward, done, False, {
            "attempt": self.attempt,
            "green": green_area,
            "action_applied": self.control_state
        }

    def _capture_frame(self):
        monitor = get_window_rect(self.hwnd)
        img = np.array(self.sct.grab(monitor))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = frame_bgr.shape[:2]
        x1_ai = int(w * 0.20)
        x2_ai = int(w * 0.90)
        frame_color_cropped = frame_bgr[:, x1_ai:x2_ai]
        return frame_bgr, frame_color_cropped

    def _get_obs(self, frame_color_cropped, original_width, bbox_abs):
        lower_white = np.array([220, 220, 220], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        mask_white = cv2.inRange(frame_color_cropped, lower_white, upper_white)

        kernel_ai = np.ones((2, 2), np.uint8)
        vision_clean = cv2.dilate(mask_white, kernel_ai, iterations=1)

        if bbox_abs is not None:
            x_abs, y_abs, bw, bh = bbox_abs
            x1_ai = int(original_width * 0.20)
            p_x1 = x_abs - x1_ai
            p_y1 = y_abs
            if p_x1 >= 0:
                cv2.rectangle(vision_clean, (p_x1, p_y1), (p_x1 + bw, p_y1 + bh), 255, -1)

        resized = cv2.resize(vision_clean, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1).astype(np.uint8)

    def close(self):
        if self.is_pressing:
            pydirectinput.keyUp("space")
        self.sct.close()


# --- 4. CALLBACK ---
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

            self.history.append({
                'episode': self.episode_count,
                'time_alive': time_alive,
                'reward': self.ep_reward,
                'loss': loss
            })

            self.ep_reward = 0.0
            self.ep_start_time = time.time()

        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            model_name = f"{self.save_path}_{self.num_timesteps}_steps"
            self.model.save(model_name)
            save_plots(self.history, show=False)
            print(f"💾 [AUTO-SAVE] Paso {self.num_timesteps}")

        return True


# --- 5. REANUDAR ENTRENAMIENTO ---
if __name__ == "__main__":
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    env = DummyVecEnv([lambda: GDEnv()])
    env = VecFrameStack(env, n_stack=4)

    os.makedirs("modelos_guardados", exist_ok=True)
    save_callback = MetricsAndSaveCallback(
        save_freq=SAVE_FREQ,
        save_path="modelos_guardados/nave_dqn_2"
    )

    print(f"📂 Fine-tuning desde: {CHECKPOINT_RESUME}")
    params_fine = {
        "learning_rate": 1.5e-5,
        "exploration_initial_eps": 0.05,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.2,
    }
    model = DQN.load(
        CHECKPOINT_RESUME,
        env=env,
        custom_objects=params_fine
    )

    # El replay buffer no se persiste en model.save(), arranca vacío.
    # Con learning_starts=0 el agente aprende desde el primer paso nuevo
    # sin esperar a que el buffer alcance un mínimo artificial.
    model.learning_starts = 0

    print("🚀 Reanudando entrenamiento desde 760k pasos.")
    print("   Configuración activa:")
    print("   · Buffer retroactivo: muerte en frame de colisión real")
    print("   · Edge penalty: y < 0.27 o y > 0.73")
    print(f"   · MIN_ACTION_HOLD_STEPS = {MIN_ACTION_HOLD_STEPS}")
    print(f"   · Guardado automático cada {SAVE_FREQ} pasos")
    print("   · Checkpoints: nave_dqn_2_800000_steps, _840000_steps ...")
    print("   Deja el juego en primer plano y no muevas el ratón...\n")

    # total_timesteps=300_000 → 760k + 300k = 1.06M (objetivo original del fichero)
    # reset_num_timesteps=False → el contador sigue desde 760k,
    # los checkpoints reflejan el total acumulado real de entrenamiento.
    model.learn(
        total_timesteps=120000,
        callback=save_callback,
        reset_num_timesteps=False
    )

    print("\n📊 Entrenamiento finalizado.")
    model.save("modelos_guardados/nave_dqn_FINAL_2")
    print("Modelo 'nave_dqn_FINAL_2' guardado.")
    save_plots(save_callback.history, show=True)
    env.close()