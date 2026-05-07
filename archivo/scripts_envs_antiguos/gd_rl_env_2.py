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

# --- 1. CONFIGURACIÓN Y CONSTANTES ---
GAME_TITLE = "Geometry Dash"
PLAYER_X_REL = 0.37
PLAYER_BAND_W = 0.06
LOWER_GREEN = np.array([46, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([46, 255, 255], dtype=np.uint8)
GREEN_AREA_MIN = 100

DEATH_FRAMES_NEEDED = 3  # Frames consecutivos sin verde para confirmar muerte (~0.2s a 30fps)
STEP_DURATION = 1 / 15   # Duración fija de cada step (15fps)
MIN_EPISODE_GAP = 1.2

METRICS_PATH = "gd_metrics.csv"
SAVE_FREQ = 10000

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
    return player_center_y, green_area

def save_plots(history, show=False):
    if not history: return
    df = pd.DataFrame(history)
    window = min(30, len(df))
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 14))
    
    # Tiempo de supervivencia con moving average
    axs[0].plot(df['episode'], df['time_alive'], color='blue', alpha=0.25, linewidth=0.8)
    df['time_smooth'] = df['time_alive'].rolling(window=window, min_periods=1).mean()
    axs[0].plot(df['episode'], df['time_smooth'], color='darkblue', linewidth=2.5, label=f'Media {window} ep')
    axs[0].axhline(y=df['time_alive'].max(), color='gold', linestyle='--', alpha=0.7, label=f'Récord: {df["time_alive"].max():.1f}s')
    axs[0].set_title('Tiempo de Supervivencia por Episodio')
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
    plt.savefig('metrics_plot_12-3.png', dpi=120)
    df.to_csv(METRICS_PATH, index=False)
    
    if show:
        plt.show() # Solo mostramos la ventana emergente al final del todo
        
    plt.close(fig)

# --- 3. CLASE DEL ENTORNO (El Gimnasio) ---
class GDEnv(gym.Env):
    def __init__(self):
        super(GDEnv, self).__init__()
        # Espacio de observación en escala de grises (1 solo canal de color)
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
        self.last_action_time = time.perf_counter()  # CAMBIO: perf_counter en lugar de time()
        self._no_green_count = 0                     # CAMBIO: sustituye player_last_seen + alive
        self._ep_frames = 0
        self._best_frames = 0
        self.last_episode_time = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._no_green_count = 0  # CAMBIO: resetear contador al inicio de cada episodio
        self._ep_frames = 0

        # Bloquear hasta que el jugador reaparezca — absorbe el tiempo muerto del respawn
        while True:
            frame = self._capture_frame()
            _, green_area = detect_player_and_band(frame)

            ahora = time.perf_counter()
            cooldown_ok = (ahora - self.last_episode_time) >= MIN_EPISODE_GAP
            if green_area >= GREEN_AREA_MIN and cooldown_ok:
                self.last_episode_time = ahora
                self.attempt += 1
                print(f"Episodio {self.attempt} iniciado. ¡Jugador detectado! Green area: {green_area}")
                break
            time.sleep(STEP_DURATION)

        obs = self._get_obs(frame)
        return obs, {}

    def step(self, action):
        step_start = time.perf_counter()  # CAMBIO: inicio del step para control de framerate
        reward = 0.05
        done = False

        now = time.perf_counter()  # CAMBIO: perf_counter para mayor precisión
        if action == 1 and (now - self.last_action_time > 0.05):
            pydirectinput.keyDown("space")
            pydirectinput.keyUp("space")
            self.last_action_time = now

        frame = self._capture_frame()
        _, green_area = detect_player_and_band(frame)
        player_visible = green_area >= GREEN_AREA_MIN

        if player_visible:
            self._no_green_count = 0  # CAMBIO: jugador visible → resetear contador
            self._ep_frames += 1
            if self._ep_frames > self._best_frames:  # Bonus por sobrevivir más de 1 segundo
                reward += 0.5
        else:
            self._no_green_count += 1  # CAMBIO: acumular frames consecutivos sin verde
            if self._no_green_count >= DEATH_FRAMES_NEEDED:
                # CAMBIO: muerte por frames consecutivos, no por DEATH_TIMEOUT en segundos
                reward = -5.0
                done = True
                self._no_green_count = 0
                if self._ep_frames > self._best_frames:
                    self._best_frames = self._ep_frames
            else:
                reward = 0.0

        obs = self._get_obs(frame)

        # CAMBIO: control de framerate — esperar lo que quede hasta completar 1/30s
        elapsed = time.perf_counter() - step_start
        remaining = STEP_DURATION - elapsed
        if remaining > 0:
            time.sleep(remaining)

        return obs, reward, done, False, {"attempt": self.attempt, "green": green_area}

    def _capture_frame(self):
        monitor = get_window_rect(self.hwnd)
        img = np.array(self.sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _get_obs(self, frame=None):
        if frame is None:
            frame = self._capture_frame()
        
        h, w = frame.shape[:2]
        x1, x2 = int(w * 0.20), int(w * 0.80)
        cropped = frame[:, x1:x2]

        #desenfoque para ignorar partículas pequeñas y ruido
        blurred = cv2.GaussianBlur(cropped, (5, 5), 0)  

        # Convertimos la imagen a negro y solo los contornos serán líneas blancas
        edges = cv2.Canny(blurred, threshold1=200, threshold2=270)

        obs = cv2.resize(edges, (84, 84))

        return np.expand_dims(obs, axis=-1).astype(np.uint8)

    def close(self):
        self.sct.close()

# --- 4. CLASE DEL CALLBACK (Guardado automático y métricas) ---
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
            if hasattr(self.model, "logger"):
                loss_dict = self.model.logger.name_to_value
                if "train/loss" in loss_dict:
                    loss = loss_dict["train/loss"]

            self.history.append({
                'episode': self.episode_count,
                'time_alive': time_alive,
                'reward': self.ep_reward,
                'loss': loss
            })

            self.ep_reward = 0.0
            self.ep_start_time = time.time()

        # GUARDADO AUTOMÁTICO cada X pasos
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            model_name = f"{self.save_path}_{self.num_timesteps}_steps"
            self.model.save(model_name)
            
            # Actualizamos la imagen silenciosamente (show=False)
            save_plots(self.history, show=False)
            
            print(f"💾 [AUTO-SAVE] ¡Progreso guardado! Modelo: '{model_name}.zip' | Gráfica actualizada. (Paso {self.num_timesteps})")

        return True

# --- 5. BUCLE PRINCIPAL DE ENTRENAMIENTO ---
if __name__ == "__main__":
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    # 1. Creamos entorno y aplicamos FRAME STACKING (Apilamos los últimos 4 frames)
    env = DummyVecEnv([lambda: GDEnv()])
    env = VecFrameStack(env, n_stack=4)
    
    # 2. Inicializamos nuestro Callback de autoguardado (cada 10,000 pasos)
    save_callback = MetricsAndSaveCallback(save_freq=SAVE_FREQ, save_path="modelos_guardados/gd_dqn")
    
    # Creamos la carpeta de modelos si no existe
    os.makedirs("modelos_guardados", exist_ok=True)

    # 3. Configuramos la IA para el aprendizaje profundo
    model = DQN("CnnPolicy", env, verbose=0, 
                buffer_size=50000, 
                learning_starts=10000,
                train_freq=4, 
                target_update_interval=2000, 
                learning_rate=5e-5,
                exploration_fraction=0.3)
    
    print("🚀 Iniciando entrenamiento PROFUNDO.")
    print(f"⚠️  Se realizarán 500,000 pasos. Guardado automático cada {SAVE_FREQ} pasos.")
    print("   Deja el juego en primer plano y no muevas el ratón...")
    
    model.learn(total_timesteps=500000, callback=save_callback)
    
    # 5. AL TERMINAR
    print("\n📊 Entrenamiento finalizado por completo.")
    
    model.save("gd_dqn_FINAL_2")
    print("✅ Modelo final guardado con éxito como 'gd_dqn_FINAL_2.zip'")

    save_plots(save_callback.history, show=True)
        
    env.close()