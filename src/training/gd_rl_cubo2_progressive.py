import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

"""
gd_rl_cubo2_progressive.py
Entrena la segunda sección de cubo usando Progressive Neural Networks
para eliminar por completo el olvido catastrófico.

Arquitectura Progressive Networks (Rusu et al., 2016):
═══════════════════════════════════════════════════════

  Input ──┬──► [Col1 Conv1] ──► [Col1 Conv2] ──► [Col1 Conv3] ──► [Col1 FC]   ← CONGELADA
          │         │                 │                │              │
          │    [Adapter 2]       [Adapter 3]      [Adapter FC]       │
          │         ↓                 ↓                ↓              │
          └──► [Col2 Conv1] ──► [Col2 Conv2] ──► [Col2 Conv3] ──► [Col2 FC] ──► Q-values
                                 (+ lateral)      (+ lateral)      (+ lateral)

  - Columna 1: pesos del cubo 1, completamente congelados (requires_grad=False).
  - Columna 2: nueva red que aprende cubo 2, inicializada desde columna 1.
  - Adapters laterales: transforman las activaciones de la columna 1 para inyectarlas
    en la columna 2. Inicializados a cero → al principio la columna 2 se comporta
    exactamente como la columna 1. Gradualmente aprenden a aprovechar las features
    congeladas del cubo 1.

Ventaja clave: NO hay olvido catastrófico porque los pesos del cubo 1 nunca se tocan.
La columna 2 puede adaptarse libremente al cubo 2 mientras mantiene acceso a las
representaciones del cubo 1 a través de las conexiones laterales.

Flujo de ejecución:
  1. Carga el modelo entrenado del cubo 1.
  2. Construye la red progresiva con columna 1 congelada.
  3. Entrena columna 2 + adapters en el cubo 2.
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
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

# ← CAMBIAR a la ruta del modelo final del cubo 1
MODELO_CUBO1 = "models/gd_dqn_FINAL_4.zip"

GAME_TITLE = "Geometry Dash"
PLAYER_X_REL = 0.345
PLAYER_BAND_W = 0.065
LOWER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
GREEN_AREA_MIN = 700

DEATH_FRAMES_NEEDED = 2
STEP_DURATION = 1 / 15
MIN_EPISODE_GAP = 1.5

METRICS_PATH = "cubo2_progressive_metrics.csv"
SAVE_FREQ = 25_000
TOTAL_STEPS = 600_000
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


def save_plots(history, show=False):
    if not history:
        return
    df = pd.DataFrame(history)
    window = min(30, len(df))

    # 4 paneles: tiempo, reward, loss, escalas laterales
    fig, axs = plt.subplots(4, 1, figsize=(12, 18))

    axs[0].plot(df['episode'], df['time_alive'], color='blue', alpha=0.25, linewidth=0.8)
    df['time_smooth'] = df['time_alive'].rolling(window=window, min_periods=1).mean()
    axs[0].plot(df['episode'], df['time_smooth'], color='darkblue', linewidth=2.5,
                label=f'Media {window} ep')
    axs[0].axhline(y=df['time_alive'].max(), color='gold', linestyle='--', alpha=0.7,
                   label=f'Récord: {df["time_alive"].max():.1f}s')
    axs[0].set_title('Tiempo de Supervivencia (Cubo 2 — Progressive Networks)')
    axs[0].set_ylabel('Segundos')
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    axs[1].plot(df['episode'], df['reward'], color='green', alpha=0.25, linewidth=0.8)
    df['reward_smooth'] = df['reward'].rolling(window=window, min_periods=1).mean()
    axs[1].plot(df['episode'], df['reward_smooth'], color='darkgreen', linewidth=2.5)
    axs[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axs[1].set_title('Recompensa Total')
    axs[1].set_ylabel('Reward')
    axs[1].grid(alpha=0.3)

    axs[2].plot(df['episode'], df['loss'], color='red', alpha=0.25, linewidth=0.8)
    df['loss_smooth'] = df['loss'].rolling(window=window, min_periods=1).mean()
    axs[2].plot(df['episode'], df['loss_smooth'], color='darkred', linewidth=2.5)
    axs[2].set_title('Pérdida DQN')
    axs[2].set_ylabel('Huber Loss')
    axs[2].grid(alpha=0.3)

    # Panel 4: evolucion de las tres escalas laterales aprendibles.
    # Si quedan en 0 indica deadlock de gradiente; si crecen,
    # significa que la columna 2 esta usando la senal de la columna 1.
    if {'lat_scale_2', 'lat_scale_3', 'lat_scale_fc'}.issubset(df.columns):
        axs[3].plot(df['episode'], df['lat_scale_2'], color='tab:purple',
                    alpha=0.6, linewidth=1.3, label='conv2 (lat_scale_2)')
        axs[3].plot(df['episode'], df['lat_scale_3'], color='tab:orange',
                    alpha=0.6, linewidth=1.3, label='conv3 (lat_scale_3)')
        axs[3].plot(df['episode'], df['lat_scale_fc'], color='tab:brown',
                    alpha=0.6, linewidth=1.3, label='fc (lat_scale_fc)')
        axs[3].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axs[3].set_title('Escalas laterales (intensidad de la conexion col1→col2)')
        axs[3].set_ylabel('Valor del escalar')
        axs[3].set_xlabel('Episodio')
        axs[3].legend(loc='best', fontsize=9)
        axs[3].grid(alpha=0.3)
    else:
        axs[3].set_visible(False)

    plt.tight_layout()
    plt.savefig('cubo2_progressive_plot.png', dpi=120)
    df.to_csv(METRICS_PATH, index=False)

    if show:
        plt.show()
    plt.close(fig)


# ─── ENTORNO ──────────────────────────────────────────────────────────────────
# (Idéntico al de tu script EWC — misma detección, mismas recompensas)

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
                print(f"[C2-PROG] Episodio {self.attempt} — green: {green_area}")
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


# ─── PROGRESSIVE NEURAL NETWORK ──────────────────────────────────────────────

class ProgressiveCNN(BaseFeaturesExtractor):
    """
    Feature extractor con arquitectura Progressive Networks.

    Dos columnas CNN con la misma estructura que NatureCNN de SB3:
      - Columna 1 (col1_*): congelada, contiene los pesos del cubo 1.
      - Columna 2 (col2_*): entrenable, aprende el cubo 2.
      - Adapters laterales (lateral_*): conexiones de col1 → col2,
        inicializadas a cero para que col2 empiece comportándose igual que col1.

    Dimensiones internas (para input 84×84 con 4 frames apilados):
      conv1: (4, 32, k=8, s=4) → 32×20×20
      conv2: (32, 64, k=4, s=2) → 64×9×9
      conv3: (64, 64, k=3, s=1) → 64×7×7
      flatten: 64*7*7 = 3136
      linear: 3136 → 512
    """

    def __init__(self, observation_space, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        n_ch = observation_space.shape[0]  # 4 con VecFrameStack(n_stack=4)

        # ── Columna 1: CONGELADA (se carga después desde el modelo cubo 1) ───
        self.col1_conv1 = nn.Conv2d(n_ch, 32, kernel_size=8, stride=4)
        self.col1_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.col1_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.col1_linear = nn.Linear(3136, features_dim)

        # ── Columna 2: ENTRENABLE (inicializada desde col1 después) ──────────
        self.col2_conv1 = nn.Conv2d(n_ch, 32, kernel_size=8, stride=4)
        self.col2_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.col2_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.col2_linear = nn.Linear(3136, features_dim)

        # ── Conexiones laterales (adapters col1 → col2) ─────────────────────
        # Mismas dimensiones de kernel/stride que la capa destino para que
        # las dimensiones espaciales coincidan al sumar.
        #
        # lateral_2: toma salida de col1_conv1 (32×20×20), produce (64×9×9)
        # lateral_3: toma salida de col1_conv2 (64×9×9),  produce (64×7×7)
        # lateral_fc: toma col1_conv3 aplanada (3136),    produce (512)
        self.lateral_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.lateral_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.lateral_fc = nn.Linear(3136, features_dim)

        # Escala aprendible por adapter para controlar cuánta señal lateral
        # entra en la columna 2. IMPORTANTE: NO se inicializan a 0, porque
        # combinado con adapters tambien a 0 (ver _init_lateral_weights)
        # provoca un punto fijo del gradiente — ambos componentes se
        # bloquean mutuamente y nunca se mueven del cero. Inicializandolas
        # a un valor pequeno positivo (~0.01) la columna 2 sigue
        # comportandose practicamente como una NatureCNN normal en el
        # primer forward, pero el gradiente fluye y el optimizador puede
        # subirlas si la senal lateral resulta util.
        self.lateral_scale_2  = nn.Parameter(th.tensor(0.01))
        self.lateral_scale_3  = nn.Parameter(th.tensor(0.01))
        self.lateral_scale_fc = nn.Parameter(th.tensor(0.01))

        # Inicializar adapters con pesos pequenos no-cero para que
        # acompanen a las escalas y permitan flujo de gradiente.
        self._init_lateral_weights()

    def _init_lateral_weights(self):
        """Inicializa los adapters laterales con pesos pequenos pero
        NO cero, para evitar el deadlock de gradiente con las escalas.

        kaiming_normal_ con escala reducida (multiplicar por 0.1 a
        posteriori) deja un magnitud comparable al ruido pero permite
        que el gradiente fluya desde el primer forward. Los biases si
        se dejan a cero — solo necesitamos romper la simetria en una
        de las dos componentes (escalas o pesos).
        """
        for module in [self.lateral_2, self.lateral_3, self.lateral_fc]:
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.weight.data.mul_(0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def load_column1(self, cnn_state_dict: dict):
        """
        Carga los pesos de NatureCNN del cubo 1 en la columna 1.

        Mapping de claves SB3 NatureCNN → columna 1:
          cnn.0.* → col1_conv1.*
          cnn.2.* → col1_conv2.*
          cnn.4.* → col1_conv3.*
          linear.0.* → col1_linear.*
        """
        mapping = {
            'cnn.0.weight': 'col1_conv1.weight', 'cnn.0.bias': 'col1_conv1.bias',
            'cnn.2.weight': 'col1_conv2.weight', 'cnn.2.bias': 'col1_conv2.bias',
            'cnn.4.weight': 'col1_conv3.weight', 'cnn.4.bias': 'col1_conv3.bias',
            'linear.0.weight': 'col1_linear.weight', 'linear.0.bias': 'col1_linear.bias',
        }
        new_state = {}
        for old_key, new_key in mapping.items():
            if old_key in cnn_state_dict:
                new_state[new_key] = cnn_state_dict[old_key]
            else:
                raise KeyError(f"Clave '{old_key}' no encontrada en el modelo del cubo 1. "
                               f"Claves disponibles: {list(cnn_state_dict.keys())}")

        # Cargamos solo las claves de col1 (strict=False para no tocar col2/laterals)
        self.load_state_dict(new_state, strict=False)

        # ── CONGELAR columna 1 ──
        for param in [self.col1_conv1, self.col1_conv2, self.col1_conv3, self.col1_linear]:
            for p in param.parameters():
                p.requires_grad = False

        print("[PROG] Columna 1 cargada y congelada.")

    def init_column2_from_column1(self):
        """
        Copia los pesos de la columna 1 a la columna 2 como punto de partida.
        Así la columna 2 arranca sabiendo jugar al cubo 1 — solo necesita
        aprender las diferencias del cubo 2.
        """
        self.col2_conv1.load_state_dict(self.col1_conv1.state_dict())
        self.col2_conv2.load_state_dict(self.col1_conv2.state_dict())
        self.col2_conv3.load_state_dict(self.col1_conv3.state_dict())
        self.col2_linear.load_state_dict(self.col1_linear.state_dict())
        print("[PROG] Columna 2 inicializada desde columna 1.")

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # ── Columna 1: forward sin gradientes (congelada) ────────────────────
        with th.no_grad():
            h1_1 = F.relu(self.col1_conv1(observations))   # (B, 32, 20, 20)
            h1_2 = F.relu(self.col1_conv2(h1_1))           # (B, 64, 9, 9)
            h1_3 = F.relu(self.col1_conv3(h1_2))           # (B, 64, 7, 7)
            h1_flat = h1_3.flatten(start_dim=1)             # (B, 3136)

        # ── Columna 2: forward con conexiones laterales ──────────────────────
        # Capa 1: sin lateral (ambas columnas ven el mismo input)
        h2_1 = F.relu(self.col2_conv1(observations))       # (B, 32, 20, 20)

        # Capa 2: col2_conv2(h2_1) + adapter(h1_1) * escala
        lat_2 = self.lateral_2(h1_1) * self.lateral_scale_2
        h2_2 = F.relu(self.col2_conv2(h2_1) + lat_2)       # (B, 64, 9, 9)

        # Capa 3: col2_conv3(h2_2) + adapter(h1_2) * escala
        lat_3 = self.lateral_3(h1_2) * self.lateral_scale_3
        h2_3 = F.relu(self.col2_conv3(h2_2) + lat_3)       # (B, 64, 7, 7)

        h2_flat = h2_3.flatten(start_dim=1)                 # (B, 3136)

        # FC: col2_linear(h2_flat) + adapter(h1_flat) * escala
        lat_fc = self.lateral_fc(h1_flat) * self.lateral_scale_fc
        output = F.relu(self.col2_linear(h2_flat) + lat_fc) # (B, 512)

        return output


# ─── CALLBACK ─────────────────────────────────────────────────────────────────

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

            # Log de escalas laterales para monitorizar cuánto se usan
            extractor = self.model.policy.q_net.features_extractor
            scales = {
                'lat_scale_2': extractor.lateral_scale_2.item(),
                'lat_scale_3': extractor.lateral_scale_3.item(),
                'lat_scale_fc': extractor.lateral_scale_fc.item(),
            }

            self.history.append({
                'episode': self.episode_count,
                'time_alive': time_alive,
                'reward': self.ep_reward,
                'loss': loss,
                **scales,
            })

            if self.episode_count % 20 == 0:
                print(f"  Escalas laterales: "
                      f"conv2={scales['lat_scale_2']:.4f}  "
                      f"conv3={scales['lat_scale_3']:.4f}  "
                      f"fc={scales['lat_scale_fc']:.4f}")

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

    # ── 1. Crear entorno ─────────────────────────────────────────────────────
    env = DummyVecEnv([lambda: GDEnvCubo()])
    env = VecFrameStack(env, n_stack=4)

    # ── 2. Cargar modelo del cubo 1 y extraer sus pesos ──────────────────────
    print("=" * 60)
    print("  CARGANDO MODELO DEL CUBO 1")
    print("=" * 60)
    print(f"  Modelo: {MODELO_CUBO1}")

    cube1_model = DQN.load(MODELO_CUBO1, env=env, device="auto")

    # Extraer pesos del feature extractor (NatureCNN)
    cube1_cnn_state = {
        k: v.clone()
        for k, v in cube1_model.policy.q_net.features_extractor.state_dict().items()
    }

    del cube1_model
    th.cuda.empty_cache() if th.cuda.is_available() else None
    print("  Pesos del cubo 1 extraídos.\n")

    # ── 3. Crear DQN con Progressive Feature Extractor ───────────────────────
    print("=" * 60)
    print("  CONSTRUYENDO RED PROGRESIVA")
    print("=" * 60)

    model = DQN(
        "CnnPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=ProgressiveCNN,
            features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
        ),
        verbose=1,
        buffer_size=50_000,
        learning_starts=5_000,
        batch_size=64,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2_000,
        learning_rate=5e-5,       # Algo más alto que EWC — no hay riesgo de olvido
        gamma=0.99,
        exploration_fraction=0.15,
        exploration_final_eps=0.02,
        device="auto",
    )

    # ── 4. Inyectar pesos del cubo 1 ─────────────────────────────────────────
    # Tanto en q_net como en q_net_target
    for qnet in [model.q_net, model.q_net_target]:
        qnet.features_extractor.load_column1(cube1_cnn_state)
        qnet.features_extractor.init_column2_from_column1()

    # Verificar que col1 está congelada y col2 + laterals son entrenables
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    frozen = total_params - trainable
    print(f"\n  Parámetros totales:    {total_params:>10,}")
    print(f"  Entrenables (col2):    {trainable:>10,}")
    print(f"  Congelados  (col1):    {frozen:>10,}")

    # Recrear el optimizador para que solo incluya parámetros entrenables.
    # Esto es importante: evita que el optimizador mantenga estados (momentum, etc.)
    # para los parámetros congelados, ahorrando memoria.
    trainable_params = [p for p in model.policy.parameters() if p.requires_grad]
    model.policy.optimizer = th.optim.Adam(trainable_params, lr=5e-5)
    print(f"  Optimizador recreado con {len(trainable_params)} grupos de parámetros.\n")

    # ── 5. Entrenar en cubo 2 ────────────────────────────────────────────────
    print("=" * 60)
    print("  ENTRENAMIENTO EN CUBO 2 (Progressive Networks)")
    print("=" * 60)
    print("  Reposiciona el juego en el nivel del CUBO 2.")
    print(f"  Se entrenarán {TOTAL_STEPS:,} pasos.")
    print(f"  Guardado cada {SAVE_FREQ:,} pasos.")
    print()
    print("  Consejo: las escalas laterales empiezan en 0.0 y subirán")
    print("  a medida que la red aprenda a usar las features del cubo 1.")
    input("\n  Pulsa ENTER cuando el juego esté en el CUBO 2...\n")

    callback = MetricsAndSaveCallback(
        save_freq=SAVE_FREQ,
        save_path="modelos_guardados/cubo2_progressive"
    )

    model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

    print("\n[FIN] Entrenamiento completado.")
    model.save("modelos_guardados/cubo2_progressive_FINAL")
    print("Modelo guardado: modelos_guardados/cubo2_progressive_FINAL.zip")
    save_plots(callback.history, show=True)
    env.close()
