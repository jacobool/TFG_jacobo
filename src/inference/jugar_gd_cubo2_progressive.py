import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

"""
jugar_gd_cubo2_progressive.py
Evaluación del modelo Progressive Networks para Geometry Dash.

Carga el modelo entrenado con la arquitectura progresiva (columna 1 congelada
+ columna 2 entrenada + adapters laterales) y juega en tiempo real.

IMPORTANTE: Este script necesita que la clase ProgressiveCNN esté disponible
para que DQN.load() pueda reconstruir la arquitectura. La importamos directamente
del script de entrenamiento, o la definimos aquí para independencia.
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
import pydirectinput
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ─── CONFIGURACIÓN (debe coincidir con el entrenamiento) ─────────────────────

GAME_TITLE = "Geometry Dash"
PLAYER_X_REL = 0.345
PLAYER_BAND_W = 0.065
LOWER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
GREEN_AREA_MIN = 600
DEATH_FRAMES_NEEDED = 2
STEP_DURATION = 1 / 15
MIN_EPISODE_GAP = 1.5
FEATURES_DIM = 512


# ─── FUNCIONES AUXILIARES ─────────────────────────────────────────────────────

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


# ─── PROGRESSIVE CNN (debe coincidir con la del entrenamiento) ────────────────

class ProgressiveCNN(BaseFeaturesExtractor):
    """
    Réplica exacta de la arquitectura usada en el entrenamiento.
    Necesaria para que DQN.load() pueda reconstruir el modelo.
    """

    def __init__(self, observation_space, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        n_ch = observation_space.shape[0]

        # Columna 1 (congelada)
        self.col1_conv1 = nn.Conv2d(n_ch, 32, kernel_size=8, stride=4)
        self.col1_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.col1_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.col1_linear = nn.Linear(3136, features_dim)

        # Columna 2 (entrenable)
        self.col2_conv1 = nn.Conv2d(n_ch, 32, kernel_size=8, stride=4)
        self.col2_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.col2_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.col2_linear = nn.Linear(3136, features_dim)

        # Conexiones laterales
        self.lateral_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.lateral_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.lateral_fc = nn.Linear(3136, features_dim)

        # Escalas
        self.lateral_scale_2 = nn.Parameter(th.tensor(0.0))
        self.lateral_scale_3 = nn.Parameter(th.tensor(0.0))
        self.lateral_scale_fc = nn.Parameter(th.tensor(0.0))

        self._init_lateral_weights()

    def _init_lateral_weights(self):
        for module in [self.lateral_2, self.lateral_3, self.lateral_fc]:
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # load_column1 e init_column2 no se necesitan para jugar,
    # pero los incluimos para compatibilidad completa.
    def load_column1(self, cnn_state_dict: dict):
        mapping = {
            'cnn.0.weight': 'col1_conv1.weight', 'cnn.0.bias': 'col1_conv1.bias',
            'cnn.2.weight': 'col1_conv2.weight', 'cnn.2.bias': 'col1_conv2.bias',
            'cnn.4.weight': 'col1_conv3.weight', 'cnn.4.bias': 'col1_conv3.bias',
            'linear.0.weight': 'col1_linear.weight', 'linear.0.bias': 'col1_linear.bias',
        }
        new_state = {}
        for old_key, new_key in mapping.items():
            new_state[new_key] = cnn_state_dict[old_key]
        self.load_state_dict(new_state, strict=False)
        for param in [self.col1_conv1, self.col1_conv2, self.col1_conv3, self.col1_linear]:
            for p in param.parameters():
                p.requires_grad = False

    def init_column2_from_column1(self):
        self.col2_conv1.load_state_dict(self.col1_conv1.state_dict())
        self.col2_conv2.load_state_dict(self.col1_conv2.state_dict())
        self.col2_conv3.load_state_dict(self.col1_conv3.state_dict())
        self.col2_linear.load_state_dict(self.col1_linear.state_dict())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        with th.no_grad():
            h1_1 = F.relu(self.col1_conv1(observations))
            h1_2 = F.relu(self.col1_conv2(h1_1))
            h1_3 = F.relu(self.col1_conv3(h1_2))
            h1_flat = h1_3.flatten(start_dim=1)

        h2_1 = F.relu(self.col2_conv1(observations))
        lat_2 = self.lateral_2(h1_1) * self.lateral_scale_2
        h2_2 = F.relu(self.col2_conv2(h2_1) + lat_2)
        lat_3 = self.lateral_3(h1_2) * self.lateral_scale_3
        h2_3 = F.relu(self.col2_conv3(h2_2) + lat_3)
        h2_flat = h2_3.flatten(start_dim=1)
        lat_fc = self.lateral_fc(h1_flat) * self.lateral_scale_fc
        output = F.relu(self.col2_linear(h2_flat) + lat_fc)

        return output


# ─── ENTORNO (versión ligera para jugar) ──────────────────────────────────────

class GDEnv(gym.Env):
    def __init__(self):
        super().__init__()
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


# ─── BUCLE DE EVALUACIÓN ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 Cargando modelo progresivo y entorno...")

    env = DummyVecEnv([lambda: GDEnv()])
    env = VecFrameStack(env, n_stack=4)

    # Cargamos sólo los pesos de la policy (sin estado del optimizador).
    # El optimizador se guardó con col1 congelada (menos param groups) y no
    # coincide con el modelo recién creado; para inferencia no lo necesitamos.
    import zipfile, io

    MODEL_PATH = "modelos_guardados/cubo2_progressive_FINAL.zip"
    try:
        model = DQN(
            "CnnPolicy",
            env,
            policy_kwargs=dict(
                features_extractor_class=ProgressiveCNN,
                features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
            ),
            device="auto",
        )
        with zipfile.ZipFile(MODEL_PATH) as zf:
            with zf.open("policy.pth") as f:
                policy_params = th.load(io.BytesIO(f.read()), map_location=model.device)
        model.policy.load_state_dict(policy_params)
        print("✅ Modelo 'cubo2_progressive_FINAL' cargado con éxito.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        exit()

    # Mostrar escalas laterales aprendidas
    extractor = model.policy.q_net.features_extractor
    print(f"\n📊 Escalas laterales aprendidas:")
    print(f"   Conv2:  {extractor.lateral_scale_2.item():.4f}")
    print(f"   Conv3:  {extractor.lateral_scale_3.item():.4f}")
    print(f"   FC:     {extractor.lateral_scale_fc.item():.4f}")

    # Congelar col1 de nuevo (la serialización no guarda requires_grad)
    for qnet in [model.q_net, model.q_net_target]:
        ext = qnet.features_extractor
        for module in [ext.col1_conv1, ext.col1_conv2, ext.col1_conv3, ext.col1_linear]:
            for p in module.parameters():
                p.requires_grad = False

    print("\n🚀 Evaluación iniciada. Pulsa Ctrl+C para detener.")
    print("   Asegúrate de tener Geometry Dash en primer plano.")

    try:
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
    except KeyboardInterrupt:
        print("\n🛑 Evaluación detenida por el usuario.")
    finally:
        env.close()
