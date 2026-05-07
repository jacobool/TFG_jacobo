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

# --- 1. CONFIGURACIÓN EXACTA (No tocar, debe coincidir con el entreno) ---
GAME_TITLE = "Geometry Dash"
PLAYER_X_REL = 0.345
PLAYER_BAND_W = 0.065
LOWER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
GREEN_AREA_MIN = 600
DEATH_FRAMES_NEEDED = 2
STEP_DURATION = 1 / 15
MIN_EPISODE_GAP = 1.1

# --- 2. FUNCIONES AUXILIARES ---
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

# --- 3. EL ENTORNO (Versión ligera para jugar) ---
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
        self.last_action_time = time.perf_counter()
        self.last_episode_time = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        while True:
            frame_bgr, frame_gray = self._capture_frame()
            _, green_area, bbox_abs = detect_player_and_band(frame_bgr)
            ahora = time.perf_counter()
            if green_area >= GREEN_AREA_MIN and (ahora - self.last_episode_time) >= MIN_EPISODE_GAP:
                self.last_episode_time = ahora
                print("🎮 Jugando... ¡Mira cómo lo hace!")
                break
            time.sleep(STEP_DURATION)
        return self._get_obs(frame_gray, frame_bgr.shape[1], bbox_abs), {}

    def step(self, action):
        step_start = time.perf_counter() 
        done = False
        now = time.perf_counter()
        
        if action == 1 and (now - self.last_action_time > 0.05):
            pydirectinput.keyDown("space")
            pydirectinput.keyUp("space")
            self.last_action_time = now

        frame_bgr, frame_gray = self._capture_frame()
        _, green_area, bbox_abs = detect_player_and_band(frame_bgr)
        
        if green_area < GREEN_AREA_MIN:
            done = True
            
        obs = self._get_obs(frame_gray, frame_bgr.shape[1], bbox_abs if not done else None)
        
        elapsed = time.perf_counter() - step_start
        if STEP_DURATION - elapsed > 0:
            time.sleep(STEP_DURATION - elapsed)

        return obs, 0, done, False, {}

    def _capture_frame(self):
        monitor = get_window_rect(self.hwnd)
        img = np.array(self.sct.grab(monitor))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = img.shape[:2]
        x1_ai, x2_ai = int(w * 0.20), int(w * 0.90) 
        frame_gray_cropped = cv2.cvtColor(img[:, x1_ai:x2_ai], cv2.COLOR_BGRA2GRAY)
        return frame_bgr, frame_gray_cropped

    def _get_obs(self, frame_gray_cropped, original_width, bbox_abs):
        _, thresh_full = cv2.threshold(frame_gray_cropped, 205, 255, cv2.THRESH_BINARY)
        kernel_ai = np.ones((2, 2), np.uint8)
        vision_clean = cv2.dilate(thresh_full, kernel_ai, iterations=1)
        if bbox_abs is not None:
            x_abs, y_abs, bw, bh = bbox_abs
            x1_ai = int(original_width * 0.20)
            p_x1 = x_abs - x1_ai
            p_y1 = y_abs
            if p_x1 >= 0: 
                cv2.rectangle(vision_clean, (p_x1, p_y1), (p_x1 + bw, p_y1 + bh), 255, -1)
        return np.expand_dims(cv2.resize(vision_clean, (84, 84), interpolation=cv2.INTER_AREA), axis=-1).astype(np.uint8)

    def close(self):
        self.sct.close()

# --- 4. BUCLE DE EVALUACIÓN ---
if __name__ == "__main__":
    print("🤖 Cargando modelo y entorno...")
    
    # 1. Crear entorno con los mismos stacks que en entrenamiento
    env = DummyVecEnv([lambda: GDEnv()])
    env = VecFrameStack(env, n_stack=4)
    
    # 2. Cargar el modelo final
    try:
        model = DQN.load("modelos_guardados/cubo2_ewc2_FINAL", env=env)
        print("✅ Modelo 'cubo2_ewc2_FINAL' cargado con éxito.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo. ¿Está en la misma carpeta? Error: {e}")
        exit()

    print("\n🚀 Evaluación iniciada. Pulsa Ctrl+C en la consola para detenerlo.")
    print("   Asegúrate de tener Geometry Dash en primer plano.")
    
    try:
        obs = env.reset()
        while True:
            # deterministic=True obliga a la IA a usar su mejor jugada, sin azar
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
    except KeyboardInterrupt:
        print("\n🛑 Evaluación detenida por el usuario.")
    finally:
        env.close()