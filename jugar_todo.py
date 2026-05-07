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
from sb3_contrib import QRDQN
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# --- 1. CONFIGURACIÓN DEL DIRECTOR ---
GAME_TITLE = "Geometry Dash"
PLAYER_X_REL = 0.345
PLAYER_BAND_W = 0.08  
LOWER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([45, 255, 255], dtype=np.uint8)

# Umbrales del Director (Interruptor Heurístico)
GREEN_AREA_MIN = 130  
UMBRAL_AREA_SPLIT = 280  

# Constantes de la Nave
SHIP_HITBOX_WIDTH = 55   
SHIP_HITBOX_HEIGHT = 45  
SHIP_OFFSET_X = 0        
SHIP_OFFSET_Y = 10       

# Físicas unificadas
STEP_DURATION = 1 / 15  
MIN_ACTION_HOLD_STEPS = 5 

def get_window_rect(hwnd):
    rect = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    point = wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
    return {"top": point.y, "left": point.x, "width": w, "height": h}

def detect_player_and_director(frame_bgr):
    h, w = frame_bgr.shape[:2]
    player_x = int(w * PLAYER_X_REL)
    band_half = int(w * PLAYER_BAND_W / 2)
    x1 = max(0, player_x - band_half)
    x2 = min(w, player_x + band_half)
    y1, y2 = int(h * 0.08), int(h * 0.92)
    band_bgr = frame_bgr[y1:y2, x1:x2]
    
    green_area = 0
    bbox_abs = None  
    modo_actual = "DESCONOCIDO"
    
    if band_bgr.size > 0:
        band_hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(band_hsv, LOWER_GREEN, UPPER_GREEN)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = max(contours, key=cv2.contourArea) if contours else None
        
        if best_cnt is not None:
            green_area = cv2.contourArea(best_cnt)
            if green_area >= GREEN_AREA_MIN:
                bx, by, bw, bh = cv2.boundingRect(best_cnt)
                
                if green_area < UMBRAL_AREA_SPLIT:
                    modo_actual = "NAVE"
                    center_x_local = bx + bw // 2
                    center_y_local = by + bh // 2
                    ex = max(0, center_x_local - (SHIP_HITBOX_WIDTH // 2) + SHIP_OFFSET_X)
                    ey = max(0, center_y_local - (SHIP_HITBOX_HEIGHT // 2) + SHIP_OFFSET_Y)
                    bbox_abs = (x1 + ex, y1 + ey, SHIP_HITBOX_WIDTH, SHIP_HITBOX_HEIGHT)
                else:
                    modo_actual = "CUBO"
                    bbox_abs = (x1 + bx, y1 + by, bw, bh)
                
    return green_area, bbox_abs, modo_actual

# --- 2. ENTORNO DEL DIRECTOR (CORREGIDO PARA EL CUBO) ---
class GDEnvDirector(gym.Env):
    def __init__(self):
        super(GDEnvDirector, self).__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)
        self.sct = mss.mss()
        self.windows = gw.getWindowsWithTitle(GAME_TITLE)
        if not self.windows:
            raise ValueError("No Geometry Dash window found")
        self.win = self.windows[0]
        self.hwnd = self.win._hWnd
        self.win.activate()
        
        self.is_pressing = False
        self.control_state = 0
        self.control_lock_steps = 0
        self.current_mode = "DESCONOCIDO"
        
        # Necesario para replicar el "Tap" exacto del entreno del cubo
        self.last_action_time = time.perf_counter()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.is_pressing:
            pydirectinput.keyUp("space")
            self.is_pressing = False
            
        self.control_state = 0
        self.control_lock_steps = 0

        while True:
            frame_bgr, frame_color_cropped = self._capture_frame()
            green_area, bbox_abs, self.current_mode = detect_player_and_director(frame_bgr)
            if self.current_mode != "DESCONOCIDO":
                break
            time.sleep(0.1)

        return self._get_obs(frame_color_cropped, frame_bgr.shape[1], bbox_abs, self.current_mode), {}

    def step(self, action):
        step_start = time.perf_counter() 
        now = time.perf_counter()
        
        if self.current_mode == "NAVE":
            # Lógica de vuelo suave (Hold)
            desired_action = int(action)
            if self.control_lock_steps > 0:
                desired_action = self.control_state
                self.control_lock_steps -= 1
            else:
                if desired_action != self.control_state:
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
                    
        elif self.current_mode == "CUBO":
            # Lógica estricta de salto discreto (Tap) - IDÉNTICA A jugar_gd_4.py
            if int(action) == 1 and (now - self.last_action_time > 0.05):
                pydirectinput.keyDown("space")
                pydirectinput.keyUp("space")
                self.last_action_time = now
            # Si cambiamos de Nave a Cubo, hay que soltar la tecla que se quedó pillada
            if self.is_pressing:
                pydirectinput.keyUp("space")
                self.is_pressing = False

        frame_bgr, frame_color_cropped = self._capture_frame()
        _, bbox_abs, self.current_mode = detect_player_and_director(frame_bgr)
        
        done = (self.current_mode == "DESCONOCIDO")
        
        # Le pasamos el current_mode para que la visión se adapte
        obs = self._get_obs(frame_color_cropped, frame_bgr.shape[1], bbox_abs if not done else None, self.current_mode)

        elapsed = time.perf_counter() - step_start
        if STEP_DURATION - elapsed > 0:
            time.sleep(STEP_DURATION - elapsed)

        return obs, 0.0, done, False, {}

    def get_current_mode(self):
        return self.current_mode

    def _capture_frame(self):
        monitor = get_window_rect(self.hwnd)
        img = np.array(self.sct.grab(monitor))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = frame_bgr.shape[:2]
        x1_ai, x2_ai = int(w * 0.20), int(w * 0.90) 
        return frame_bgr, frame_bgr[:, x1_ai:x2_ai]

    def _get_obs(self, frame_color_cropped, original_width, bbox_abs, mode):
        # --- DOBLE VISIÓN: Respetando el entrenamiento de cada agente ---
        if mode == "CUBO":
            # Filtro antiguo de Grises > 205
            frame_gray_cropped = cv2.cvtColor(frame_color_cropped, cv2.COLOR_BGR2GRAY)
            _, vision_clean = cv2.threshold(frame_gray_cropped, 205, 255, cv2.THRESH_BINARY)
        else:
            # Filtro nuevo de la Nave RGB > 220
            lower_white = np.array([220, 220, 220], dtype=np.uint8)
            upper_white = np.array([255, 255, 255], dtype=np.uint8)
            vision_clean = cv2.inRange(frame_color_cropped, lower_white, upper_white)
        
        kernel_ai = np.ones((2, 2), np.uint8)
        vision_clean = cv2.dilate(vision_clean, kernel_ai, iterations=1)

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

# --- 3. BUCLE DEL ORQUESTADOR MULTI-AGENTE ---
if __name__ == "__main__":
    print("⏳ Iniciando el Sistema Orquestador en modo SILENCIOSO...")
    
    RUTA_MODELO_CUBO = "modelos_guardados/gd_qrdqn_replay_cubo2_600k_FINAL" # <-- Cambia esto
    RUTA_MODELO_NAVE = "modelos_guardados/nave_qrdqn_FINAL" # <-- Cambia esto
    
    try:
        model_cubo = QRDQN.load(RUTA_MODELO_CUBO)
        print("✅ Cerebro del CUBO cargado.")
        model_nave = QRDQN.load(RUTA_MODELO_NAVE)
        print("✅ Cerebro de la NAVE cargado.")
    except FileNotFoundError as e:
        print(f"❌ ERROR AL CARGAR MODELOS: {e}")
        exit()

    env_raw = GDEnvDirector()
    env = DummyVecEnv([lambda: env_raw])
    env = VecFrameStack(env, n_stack=4)

    print("🚀 SISTEMA ORQUESTADOR LISTO. ¡Entra a cualquier nivel!")
    print("⚠️ Pulsa CTRL+C en esta terminal para detener la IA.")
    
    obs = env.reset()
    try:
        while True:
            modo_actual = env_raw.get_current_mode()
            
            if modo_actual == "CUBO":
                action, _ = model_cubo.predict(obs, deterministic=True)
            elif modo_actual == "NAVE":
                action, _ = model_nave.predict(obs, deterministic=True)
            else:
                action = [0] 

            obs, rewards, dones, info = env.step(action)
            
    except KeyboardInterrupt:
        print("\n🛑 Orquestador detenido manualmente.")
    finally:
        env_raw.close()