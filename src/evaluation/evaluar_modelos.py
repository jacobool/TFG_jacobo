import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

"""
evaluar_modelos.py
Framework de evaluación cuantitativa para el TFG.

Ejecuta N episodios de un modelo y calcula métricas rigurosas con
intervalos de confianza. Diseñado para comparar:
  - Modelo base (solo cubo 1)
  - Modelo EWC (cubo 1 + cubo 2)
  - Modelo Progressive (cubo 1 + cubo 2)

Métricas calculadas:
  ┌─────────────────────────────────────────────────────────────┐
  │ RENDIMIENTO       │ Supervivencia, progreso, recompensa    │
  │ EFICIENCIA        │ Ratio saltos, saltos vacíos, cadencia  │
  │ CONSISTENCIA      │ σ, CV, percentiles, racha mejor/peor   │
  │ TRANSFERENCIA     │ Backward transfer, forward transfer    │
  └─────────────────────────────────────────────────────────────┘

Uso:
  python evaluar_modelos.py --model modelos_guardados/cubo2_progressive_FINAL
                            --episodes 50
                            --output resultados_progressive.csv

Para comparar modelos, ejecuta varias veces con distintos --model y luego
usa --compare para generar tablas y gráficos de comparación:
  python evaluar_modelos.py --compare resultados_base.csv resultados_ewc.csv resultados_progressive.csv
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
import sys
import argparse
import pydirectinput
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

GAME_TITLE = "Geometry Dash"
PLAYER_X_REL = 0.345
PLAYER_BAND_W = 0.065
LOWER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
GREEN_AREA_MIN = 500
DEATH_FRAMES_NEEDED = 1
STEP_DURATION = 1 / 15
MIN_EPISODE_GAP = 1.1
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


# ─── PROGRESSIVE CNN (necesaria para cargar modelos progresivos) ──────────────

class ProgressiveCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        n_ch = observation_space.shape[0]
        self.col1_conv1 = nn.Conv2d(n_ch, 32, kernel_size=8, stride=4)
        self.col1_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.col1_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.col1_linear = nn.Linear(3136, features_dim)
        self.col2_conv1 = nn.Conv2d(n_ch, 32, kernel_size=8, stride=4)
        self.col2_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.col2_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.col2_linear = nn.Linear(3136, features_dim)
        self.lateral_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.lateral_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.lateral_fc = nn.Linear(3136, features_dim)
        self.lateral_scale_2 = nn.Parameter(th.tensor(0.0))
        self.lateral_scale_3 = nn.Parameter(th.tensor(0.0))
        self.lateral_scale_fc = nn.Parameter(th.tensor(0.0))
        self._init_lateral_weights()

    def _init_lateral_weights(self):
        for module in [self.lateral_2, self.lateral_3, self.lateral_fc]:
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def load_column1(self, s):
        pass

    def init_column2_from_column1(self):
        pass

    def forward(self, x):
        with th.no_grad():
            h1_1 = F.relu(self.col1_conv1(x))
            h1_2 = F.relu(self.col1_conv2(h1_1))
            h1_3 = F.relu(self.col1_conv3(h1_2))
            h1_flat = h1_3.flatten(start_dim=1)
        h2_1 = F.relu(self.col2_conv1(x))
        h2_2 = F.relu(self.col2_conv2(h2_1) + self.lateral_2(h1_1) * self.lateral_scale_2)
        h2_3 = F.relu(self.col2_conv3(h2_2) + self.lateral_3(h1_2) * self.lateral_scale_3)
        h2_flat = h2_3.flatten(start_dim=1)
        return F.relu(self.col2_linear(h2_flat) + self.lateral_fc(h1_flat) * self.lateral_scale_fc)


# ─── ENTORNO DE EVALUACIÓN ───────────────────────────────────────────────────
# Versión instrumentada que registra métricas detalladas por step.

class GDEnvEval(gym.Env):
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
        self.last_action_time = time.perf_counter()
        self.last_episode_time = 0.0

        # ── Métricas por episodio ──
        self._ep_start = 0.0
        self._ep_frames = 0
        self._ep_jumps = 0
        self._ep_jump_times = []       # timestamps de cada salto
        self._no_green_count = 0
        self._ep_reward = 0.0
        self._player_y_history = []    # posición Y del jugador (para analizar trayectoria)
        self._last_episode_metrics = None  # snapshot que sobrevive al reset

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._no_green_count = 0
        self._ep_frames = 0
        self._ep_jumps = 0
        self._ep_jump_times = []
        self._ep_reward = 0.0
        self._player_y_history = []

        while True:
            frame_bgr, frame_gray = self._capture_frame()
            player_y, green_area, bbox_abs = detect_player_and_band(frame_bgr)
            ahora = time.perf_counter()
            if green_area >= GREEN_AREA_MIN and (ahora - self.last_episode_time) >= MIN_EPISODE_GAP:
                self.last_episode_time = ahora
                self._ep_start = ahora
                break
            time.sleep(STEP_DURATION)
        return self._get_obs(frame_gray, frame_bgr.shape[1], bbox_abs), {}

    def step(self, action):
        step_start = time.perf_counter()
        done = False
        now = time.perf_counter()

        # Registrar acción
        if action == 1 and (now - self.last_action_time > 0.05):
            pydirectinput.keyDown("space")
            pydirectinput.keyUp("space")
            self.last_action_time = now
            self._ep_jumps += 1
            self._ep_jump_times.append(now - self._ep_start)

        frame_bgr, frame_gray = self._capture_frame()
        player_y, green_area, bbox_abs = detect_player_and_band(frame_bgr)

        if green_area >= GREEN_AREA_MIN:
            self._no_green_count = 0
            self._ep_frames += 1
            self._player_y_history.append(player_y)
        else:
            self._no_green_count += 1
            if self._no_green_count >= DEATH_FRAMES_NEEDED:
                done = True
                self._no_green_count = 0
                # ── CLAVE: guardar snapshot ANTES de que DummyVecEnv llame reset() ──
                self._last_episode_metrics = self._compute_metrics()

        obs = self._get_obs(frame_gray, frame_bgr.shape[1], bbox_abs if not done else None)

        elapsed = time.perf_counter() - step_start
        if STEP_DURATION - elapsed > 0:
            time.sleep(STEP_DURATION - elapsed)

        return obs, 0, done, False, {}

    def _compute_metrics(self):
        """Calcula las métricas del episodio actual (llamado internamente)."""
        duration = time.perf_counter() - self._ep_start
        frames = max(self._ep_frames, 1)

        # Cadencia de salto (saltos por segundo)
        jumps_per_sec = self._ep_jumps / max(duration, 0.01)

        # Regularidad de salto: CV de intervalos entre saltos consecutivos
        jump_intervals = np.diff(self._ep_jump_times) if len(self._ep_jump_times) > 1 else []
        jump_regularity = float(np.std(jump_intervals) / np.mean(jump_intervals)) if len(jump_intervals) > 1 else 0.0

        # Variabilidad vertical del jugador (cuánto oscila en Y)
        y_std = float(np.std(self._player_y_history)) if len(self._player_y_history) > 2 else 0.0

        return {
            'duration_s': round(duration, 3),
            'frames': frames,
            'total_jumps': self._ep_jumps,
            'jumps_per_sec': round(jumps_per_sec, 3),
            'jump_regularity_cv': round(jump_regularity, 3),
            'player_y_std': round(y_std, 2),
        }

    def get_episode_metrics(self):
        """Devuelve las métricas del último episodio completado (snapshot)."""
        return self._last_episode_metrics

    def _capture_frame(self):
        monitor = get_window_rect(self.hwnd)
        img = np.array(self.sct.grab(monitor))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = img.shape[:2]
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
        return np.expand_dims(cv2.resize(vision_clean, (84, 84), interpolation=cv2.INTER_AREA), axis=-1).astype(np.uint8)

    def close(self):
        self.sct.close()


# ─── FUNCIONES DE ANÁLISIS ────────────────────────────────────────────────────

def compute_summary_stats(df, label="Modelo"):
    """Calcula estadísticos resumen con intervalos de confianza al 95%."""
    n = len(df)
    results = {'modelo': label, 'n_episodios': n}

    for col in ['duration_s', 'frames', 'total_jumps', 'jumps_per_sec']:
        data = df[col].values
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        se = std / np.sqrt(n)

        # IC 95% con t-student
        if n > 1:
            t_crit = stats.t.ppf(0.975, df=n - 1)
            ci_low = mean - t_crit * se
            ci_high = mean + t_crit * se
        else:
            ci_low = ci_high = mean

        results[f'{col}_mean'] = round(mean, 3)
        results[f'{col}_std'] = round(std, 3)
        results[f'{col}_median'] = round(np.median(data), 3)
        results[f'{col}_ci95_low'] = round(ci_low, 3)
        results[f'{col}_ci95_high'] = round(ci_high, 3)
        results[f'{col}_p25'] = round(np.percentile(data, 25), 3)
        results[f'{col}_p75'] = round(np.percentile(data, 75), 3)

    # CV (coeficiente de variación) de supervivencia → consistencia
    dur = df['duration_s'].values
    results['consistency_cv'] = round(np.std(dur, ddof=1) / max(np.mean(dur), 0.01), 3)

    # Mejor y peor racha (rachas de episodios sobre/bajo la mediana)
    median_dur = np.median(dur)
    above = dur >= median_dur
    best_streak = worst_streak = current = 0
    for v in above:
        if v:
            current += 1
            best_streak = max(best_streak, current)
        else:
            current = 0
    current = 0
    for v in ~above:
        if v:
            current += 1
            worst_streak = max(worst_streak, current)
        else:
            current = 0
    results['best_streak'] = best_streak
    results['worst_streak'] = worst_streak

    return results


def compute_transfer_metrics(df_before, df_after, task_name="cubo1"):
    """
    Calcula métricas de transferencia.

    Backward Transfer (BT):
      BT = media_después - media_antes
      BT < 0 → olvido catastrófico
      BT ≈ 0 → sin olvido
      BT > 0 → mejora (raro pero posible)

    Se incluye un test estadístico (Welch's t-test) para determinar
    si la diferencia es significativa.
    """
    before = df_before['duration_s'].values
    after = df_after['duration_s'].values

    bt = np.mean(after) - np.mean(before)
    bt_relative = bt / max(np.mean(before), 0.01) * 100  # en porcentaje

    # Welch's t-test (no asume varianzas iguales)
    if len(before) > 1 and len(after) > 1:
        t_stat, p_value = stats.ttest_ind(after, before, equal_var=False)
    else:
        t_stat, p_value = 0.0, 1.0

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(before, ddof=1) ** 2 + np.std(after, ddof=1) ** 2) / 2)
    cohens_d = bt / max(pooled_std, 0.01)

    return {
        'task': task_name,
        'mean_before': round(np.mean(before), 3),
        'mean_after': round(np.mean(after), 3),
        'backward_transfer': round(bt, 3),
        'backward_transfer_pct': round(bt_relative, 1),
        't_statistic': round(t_stat, 3),
        'p_value': round(p_value, 4),
        'cohens_d': round(cohens_d, 3),
        'significant': p_value < 0.05,
        'interpretation': (
            "Sin olvido significativo" if p_value >= 0.05
            else ("Olvido catastrófico" if bt < 0 else "Mejora significativa")
        ),
    }


def generate_comparison_plots(csv_files, output_path="comparacion_modelos.png"):
    """Genera gráficos comparativos entre múltiples modelos."""
    dfs = {}
    for f in csv_files:
        label = os.path.splitext(os.path.basename(f))[0].replace("resultados_", "")
        dfs[label] = pd.read_csv(f)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#378ADD', '#1D9E75', '#D85A30', '#D4537E']

    # 1. Box plot: supervivencia
    ax = axs[0, 0]
    data_dur = [dfs[k]['duration_s'].values for k in dfs]
    bp = ax.boxplot(data_dur, labels=list(dfs.keys()), patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(dfs)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    ax.set_ylabel('Segundos')
    ax.set_title('Supervivencia por modelo')

    # 2. Box plot: saltos por segundo
    ax = axs[0, 1]
    data_jps = [dfs[k]['jumps_per_sec'].values for k in dfs]
    bp = ax.boxplot(data_jps, labels=list(dfs.keys()), patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(dfs)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    ax.set_ylabel('Saltos/s')
    ax.set_title('Cadencia de salto')

    # 3. Distribución de supervivencia (histograma + KDE)
    ax = axs[1, 0]
    for (label, df), color in zip(dfs.items(), colors):
        dur = df['duration_s'].values
        ax.hist(dur, bins=15, alpha=0.3, color=color, label=label, density=True)
        if len(dur) > 3:
            kde = stats.gaussian_kde(dur)
            x_range = np.linspace(dur.min(), dur.max(), 100)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2)
    ax.set_xlabel('Supervivencia (s)')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribución de supervivencia')
    ax.legend()

    # 4. Tabla resumen
    ax = axs[1, 1]
    ax.axis('off')
    table_data = []
    headers = ['Modelo', 'Media (s)', 'Mediana', 'σ', 'CV', 'Saltos/s']
    for label, df in dfs.items():
        dur = df['duration_s']
        jps = df['jumps_per_sec']
        table_data.append([
            label,
            f"{dur.mean():.2f}",
            f"{dur.median():.2f}",
            f"{dur.std():.2f}",
            f"{dur.std() / max(dur.mean(), 0.01):.2f}",
            f"{jps.mean():.2f}",
        ])
    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Colorear encabezados
    for j, header in enumerate(headers):
        table[(0, j)].set_facecolor('#EEEDFE')
        table[(0, j)].set_text_props(fontweight='bold')

    plt.suptitle('Comparación de modelos — Geometry Dash RL', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Gráfico guardado: {output_path}")
    plt.show()


# ─── EVALUACIÓN PRINCIPAL ────────────────────────────────────────────────────

def run_evaluation(model_path, n_episodes, output_csv, is_progressive=False):
    """Ejecuta N episodios y registra métricas detalladas."""

    print("=" * 60)
    print(f"  EVALUACIÓN: {os.path.basename(model_path)}")
    print(f"  Episodios: {n_episodes}")
    print("=" * 60)

    env = DummyVecEnv([lambda: GDEnvEval()])
    env = VecFrameStack(env, n_stack=4)

    # Cargar modelo (detectar si es progresivo)
    try:
        if is_progressive:
            model = DQN.load(model_path, env=env, custom_objects={
                "policy_kwargs": dict(
                    features_extractor_class=ProgressiveCNN,
                    features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
                ),
            })
            print("  Tipo: Progressive Networks")
        else:
            model = DQN.load(model_path, env=env)
            print("  Tipo: DQN estándar / EWC")
    except Exception as e:
        print(f"  ❌ Error al cargar: {e}")
        print("  Intenta con --progressive si es un modelo progresivo.")
        return

    print(f"  Modelo cargado. ¡Asegúrate de que GD está en primer plano!")
    input("\n  Pulsa ENTER para empezar la evaluación...\n")

    # Acceder al entorno real (debajo de VecFrameStack y DummyVecEnv)
    raw_env = env.envs[0]

    all_metrics = []
    try:
        for ep in range(n_episodes):
            obs = env.reset()
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, _ = env.step(action)
                done = dones[0]

            # Recoger métricas del episodio (snapshot guardado en step cuando done=True)
            ep_metrics = raw_env.get_episode_metrics()
            if ep_metrics is None:
                print(f"  Ep {ep + 1:3d}/{n_episodes} │ ⚠ Sin métricas (episodio demasiado corto)")
                continue
            ep_metrics['episode'] = ep + 1
            all_metrics.append(ep_metrics)

            # Progreso en consola
            print(f"  Ep {ep + 1:3d}/{n_episodes} │ "
                  f"Duración: {ep_metrics['duration_s']:6.2f}s │ "
                  f"Frames: {ep_metrics['frames']:4d} │ "
                  f"Saltos: {ep_metrics['total_jumps']:3d} │ "
                  f"Saltos/s: {ep_metrics['jumps_per_sec']:.2f}")

    except KeyboardInterrupt:
        print(f"\n  ⚠ Evaluación interrumpida en episodio {len(all_metrics)}.")

    env.close()

    if not all_metrics:
        print("  No se completó ningún episodio.")
        return

    # Guardar CSV con datos crudos
    df = pd.DataFrame(all_metrics)
    df.to_csv(output_csv, index=False)
    print(f"\n  💾 Datos guardados: {output_csv}")

    # Calcular y mostrar resumen
    summary = compute_summary_stats(df, label=os.path.basename(model_path))

    print("\n" + "=" * 60)
    print("  RESUMEN DE RESULTADOS")
    print("=" * 60)
    print(f"  Episodios completados:  {summary['n_episodios']}")
    print()
    print(f"  ── Supervivencia ──")
    print(f"  Media:    {summary['duration_s_mean']:.2f}s  "
          f"(IC 95%: [{summary['duration_s_ci95_low']:.2f}, {summary['duration_s_ci95_high']:.2f}])")
    print(f"  Mediana:  {summary['duration_s_median']:.2f}s")
    print(f"  σ:        {summary['duration_s_std']:.2f}s")
    print(f"  P25-P75:  [{summary['duration_s_p25']:.2f}, {summary['duration_s_p75']:.2f}]s")
    print()
    print(f"  ── Eficiencia de acción ──")
    print(f"  Saltos/s (media):  {summary['jumps_per_sec_mean']:.2f}  "
          f"(IC 95%: [{summary['jumps_per_sec_ci95_low']:.2f}, {summary['jumps_per_sec_ci95_high']:.2f}])")
    print(f"  Saltos totales (media): {summary['total_jumps_mean']:.1f}")
    print()
    print(f"  ── Consistencia ──")
    print(f"  CV (coef. variación):  {summary['consistency_cv']:.3f}  "
          f"({'Muy consistente' if summary['consistency_cv'] < 0.15 else 'Consistente' if summary['consistency_cv'] < 0.3 else 'Variable' if summary['consistency_cv'] < 0.5 else 'Muy variable'})")
    print(f"  Mejor racha:   {summary['best_streak']} episodios sobre mediana")
    print(f"  Peor racha:    {summary['worst_streak']} episodios bajo mediana")
    print("=" * 60)

    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación cuantitativa de modelos RL para Geometry Dash")
    parser.add_argument("--model", type=str, help="Ruta al modelo .zip")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Número de episodios a evaluar (default: 50)")
    parser.add_argument("--output", type=str, default="resultados.csv",
                        help="Archivo CSV de salida")
    parser.add_argument("--progressive", action="store_true",
                        help="Usar si el modelo es Progressive Networks")
    parser.add_argument("--compare", nargs='+', type=str,
                        help="Lista de CSVs para comparar modelos")
    parser.add_argument("--transfer", nargs=2, type=str, metavar=('BEFORE', 'AFTER'),
                        help="Calcular backward transfer: CSV_antes CSV_después")

    args = parser.parse_args()

    if args.compare:
        # Modo comparación
        print("📊 Generando comparación entre modelos...\n")
        generate_comparison_plots(args.compare)

    elif args.transfer:
        # Modo backward transfer
        print("📊 Calculando backward transfer...\n")
        df_before = pd.read_csv(args.transfer[0])
        df_after = pd.read_csv(args.transfer[1])
        results = compute_transfer_metrics(df_before, df_after)

        print("=" * 60)
        print("  BACKWARD TRANSFER")
        print("=" * 60)
        print(f"  Media antes:    {results['mean_before']:.2f}s")
        print(f"  Media después:  {results['mean_after']:.2f}s")
        print(f"  BT absoluto:    {results['backward_transfer']:+.2f}s")
        print(f"  BT relativo:    {results['backward_transfer_pct']:+.1f}%")
        print(f"  t-statistic:    {results['t_statistic']:.3f}")
        print(f"  p-value:        {results['p_value']:.4f}")
        print(f"  Cohen's d:      {results['cohens_d']:.3f}  "
              f"({'pequeño' if abs(results['cohens_d']) < 0.5 else 'mediano' if abs(results['cohens_d']) < 0.8 else 'grande'})")
        print(f"  ──────────────────────────────────────────")
        print(f"  Conclusión:     {results['interpretation']}")
        if results['significant'] and results['backward_transfer'] < 0:
            print(f"  ⚠ El modelo olvidó significativamente la tarea anterior.")
        elif not results['significant']:
            print(f"  ✅ No hay evidencia de olvido catastrófico (p={results['p_value']:.3f}).")
        print("=" * 60)

    elif args.model:
        # Modo evaluación
        run_evaluation(args.model, args.episodes, args.output, args.progressive)
    else:
        parser.print_help()
        print("\n── Ejemplos de uso ──")
        print()
        print("  1. Evaluar un modelo (50 episodios):")
        print("     python evaluar_modelos.py --model models/gd_dqn_FINAL_4 --episodes 50 --output resultados_base.csv")
        print()
        print("  2. Evaluar modelo progresivo:")
        print("     python evaluar_modelos.py --model modelos_guardados/cubo2_progressive_FINAL --episodes 50 --output resultados_prog.csv --progressive")
        print()
        print("  3. Comparar modelos:")
        print("     python evaluar_modelos.py --compare resultados_base.csv resultados_ewc.csv resultados_prog.csv")
        print()
        print("  4. Calcular backward transfer (¿olvidó el cubo 1?):")
        print("     python evaluar_modelos.py --transfer resultados_cubo1_antes.csv resultados_cubo1_despues.csv")
