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

# --- 1. CONFIGURACIÓN EXACTA (Debe coincidir con el entrenamiento) ---
GAME_TITLE = "Geometry Dash"
PLAYER_X_REL = 0.345
PLAYER_BAND_W = 0.08  
LOWER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
GREEN_AREA_MIN = 130  

SHIP_HITBOX_WIDTH = 55   
SHIP_HITBOX_HEIGHT = 45  
SHIP_OFFSET_X = 0        
SHIP_OFFSET_Y = 10       

# ¡IMPORTANTE! Mismo "lag" mental con el que entrenó
STEP_DURATION = 1 / 15  
MIN_ACTION_HOLD_STEPS = 4

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
            if largest_piece_area >= GREEN_AREA_MIN:
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
                
    return player_center_y, green_area, bbox_abs

# --- 2. ENTORNO DE EVALUACIÓN (Sin sistema de recompensas) ---
class GDEnvPlay(gym.Env):
    def __init__(self):
        super(GDEnvPlay, self).__init__()
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
        self.last_obs_84 = np.zeros((84, 84), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.is_pressing:
            pydirectinput.keyUp("space")
            self.is_pressing = False
        self.control_state = 0
        self.control_lock_steps = 0

        # Esperar hasta que la nave esté viva para empezar
        while True:
            frame_bgr, frame_color_cropped = self._capture_frame()
            _, green_area, bbox_abs = detect_player_and_band(frame_bgr)
            if green_area >= GREEN_AREA_MIN:
                break
            time.sleep(0.1)

        return self._get_obs(frame_color_cropped, frame_bgr.shape[1], bbox_abs), {}

    def step(self, action):
        step_start = time.perf_counter() 

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

        frame_bgr, frame_color_cropped = self._capture_frame()
        _, green_area, bbox_abs = detect_player_and_band(frame_bgr)
        done = green_area < GREEN_AREA_MIN

        obs = self._get_obs(frame_color_cropped, frame_bgr.shape[1], bbox_abs if not done else None)

        # Monitor visual para ti
        cv2.putText(frame_bgr, f"IA NAVE JUGANDO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        accion_texto = "MOTOR ENCENDIDO" if self.is_pressing else "CAYENDO"
        color_accion = (0, 255, 0) if self.is_pressing else (0, 0, 255)
        cv2.putText(frame_bgr, accion_texto, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_accion, 2)
        
        if bbox_abs is not None:
            cv2.rectangle(frame_bgr, (bbox_abs[0], bbox_abs[1]), (bbox_abs[0]+bbox_abs[2], bbox_abs[1]+bbox_abs[3]), (0, 255, 0), 2)

        # Muestra la misma entrada 84x84 que procesa la red (similar a calibrate_nave.py)
        ai_vision_bgr = cv2.cvtColor(self.last_obs_84, cv2.COLOR_GRAY2BGR)
        ai_display_size = min(252, frame_bgr.shape[0], frame_bgr.shape[1])
        ai_vision_display = cv2.resize(ai_vision_bgr, (ai_display_size, ai_display_size), interpolation=cv2.INTER_NEAREST)
        h, w = frame_bgr.shape[:2]
        frame_bgr[h-ai_display_size:h, w-ai_display_size:w] = ai_vision_display
        cv2.rectangle(frame_bgr, (w-ai_display_size, h-ai_display_size), (w, h), (255, 0, 255), 2)
        cv2.putText(frame_bgr, "Vision IA 84x84", (w-ai_display_size + 6, h-ai_display_size + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

        # Banda vertical de deteccion donde se estima la posicion de la nave
        p_x_center = int(w * PLAYER_X_REL)
        band_half = int(w * PLAYER_BAND_W / 2)
        cv2.rectangle(frame_bgr, (max(0, p_x_center - band_half), int(h * 0.08)),
                             (min(w, p_x_center + band_half), int(h * 0.92)), (0, 255, 255), 1)

        # FPS real de inferencia
        elapsed_now = time.perf_counter() - step_start
        fps = 1.0 / max(elapsed_now, 1e-4)
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("IA Jugando", frame_bgr)
        cv2.waitKey(1)

        # Respetamos el lag de la IA
        elapsed = time.perf_counter() - step_start
        if STEP_DURATION - elapsed > 0:
            time.sleep(STEP_DURATION - elapsed)

        return obs, 0.0, done, False, {}

    def _capture_frame(self):
        monitor = get_window_rect(self.hwnd)
        img = np.array(self.sct.grab(monitor))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = frame_bgr.shape[:2]
        x1_ai, x2_ai = int(w * 0.20), int(w * 0.90) 
        return frame_bgr, frame_bgr[:, x1_ai:x2_ai]

    def _get_obs(self, frame_color_cropped, original_width, bbox_abs):
        # Filtro de blancos (Modo Patata/Limpio)
        lower_white = np.array([220, 220, 220], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        vision_clean = cv2.inRange(frame_color_cropped, lower_white, upper_white)
        
        kernel_ai = np.ones((2, 2), np.uint8)
        vision_clean = cv2.dilate(vision_clean, kernel_ai, iterations=1)

        if bbox_abs is not None:
            x_abs, y_abs, bw, bh = bbox_abs
            p_x1 = x_abs - int(original_width * 0.20)
            if p_x1 >= 0:
                cv2.rectangle(vision_clean, (p_x1, y_abs), (p_x1 + bw, y_abs + bh), 255, -1)

        resized = cv2.resize(vision_clean, (84, 84), interpolation=cv2.INTER_AREA)
        self.last_obs_84 = resized
        return np.expand_dims(resized, axis=-1).astype(np.uint8)

    def close(self):
        if self.is_pressing:
            pydirectinput.keyUp("space")
        self.sct.close()
        cv2.destroyAllWindows()

# --- 3. BUCLE DE JUEGO ---
if __name__ == "__main__":
    print("⏳ Cargando el cerebro de la Nave...")
    
    # Creamos el entorno y apilamos los 4 frames igual que en el entreno
    env = DummyVecEnv([lambda: GDEnvPlay()])
    env = VecFrameStack(env, n_stack=4)
    
    # Cargar el modelo final (o cambia el nombre si quieres probar un checkpoint)
    try:
        model = DQN.load("modelos_guardados/nave_dqn_1_800000_steps")
        print("✅ Cerebro 'nave_dqn_1_800000_steps' cargado correctamente.")
    except FileNotFoundError:
        print("❌ ERROR: No se encuentra 'modelos_guardados/nave_dqn_1_800000_steps.zip'.")
        exit()

    print("🚀 La IA está lista. ¡Inicia un nivel con la nave!")
    
    obs = env.reset()
    try:
        while True:
            # deterministic=True es clave: le quita la aletoriedad de exploración. 
            # Ya no está probando cosas nuevas, va a jugar con lo que sabe 100% seguro.
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            
    except KeyboardInterrupt:
        print("\n🛑 IA detenida manualmente por el usuario.")
    finally:
        env.close()