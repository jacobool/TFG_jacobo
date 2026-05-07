import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pygetwindow as gw
import ctypes
from ctypes import wintypes
import time
import pydirectinput
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# --- CONSTANTES ---
GAME_TITLE = "Geometry Dash"
PLAYER_X_REL = 0.37
PLAYER_BAND_W = 0.06
LOWER_GREEN = np.array([46, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([46, 255, 255], dtype=np.uint8)
GREEN_AREA_MIN = 100
STEP_DURATION = 1 / 15  # Los mismos 15 FPS del entrenamiento

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
    
    green_area = 0
    if band_bgr.size > 0:
        band_hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(band_hsv, LOWER_GREEN, UPPER_GREEN)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        green_area = cv2.countNonZero(mask_clean)
    return green_area

# --- ENTORNO CLONADO (Solo para jugar) ---
class GDPlayEnv(gym.Env):
    def __init__(self):
        super(GDPlayEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)
        self.sct = mss.mss()
        self.windows = gw.getWindowsWithTitle(GAME_TITLE)
        if not self.windows:
            raise ValueError("❌ Abre Geometry Dash primero.")
        self.win = self.windows[0]
        self.hwnd = self.win._hWnd
        self.win.activate()
        self.last_action_time = time.perf_counter()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        while True:
            frame = self._capture_frame()
            green_area = detect_player_and_band(frame)
            if green_area >= GREEN_AREA_MIN:
                break
            time.sleep(STEP_DURATION)
        return self._get_obs(frame), {}

    def step(self, action):
        step_start = time.perf_counter()
        now = time.perf_counter()
        
        # Ejecutar salto si la IA lo decide
        if action == 1 and (now - self.last_action_time > 0.05):
            pydirectinput.keyDown("space")
            pydirectinput.keyUp("space")
            self.last_action_time = now

        frame = self._capture_frame()
        green_area = detect_player_and_band(frame)
        done = green_area < GREEN_AREA_MIN  # Muere si no ve verde
        
        obs = self._get_obs(frame)

        # Control exacto de 15 FPS
        elapsed = time.perf_counter() - step_start
        remaining = STEP_DURATION - elapsed
        if remaining > 0:
            time.sleep(remaining)

        return obs, 0, done, False, {}

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
        blurred = cv2.GaussianBlur(cropped, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=200, threshold2=270)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        obs = cv2.resize(dilated, (84, 84))
        return np.expand_dims(obs, axis=-1).astype(np.uint8)

# --- BUCLE DE JUEGO ---
if __name__ == "__main__":
    print("🧠 Preparando el cerebro de la IA...")
    
    # 1. Creamos el entorno y apilamos 4 frames igual que en el entrenamiento
    env = DummyVecEnv([lambda: GDPlayEnv()])
    env = VecFrameStack(env, n_stack=4)
    
    # 2. CARGAMOS EL MODELO (Pon aquí el nombre exacto de tu archivo zip)
    modelo_path = "modelos_guardados/gd_dqn_470000_steps.zip" 
    
    print(f"🔄 Cargando modelo: {modelo_path}")
    model = DQN.load(modelo_path, env=env)
    
    print("🎮 ¡IA al volante! Pon Geometry Dash en pantalla y no toques el ratón.")
    
    obs = env.reset()
    while True:
        # deterministic=True significa que no explora al azar, usa su mejor conocimiento
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)