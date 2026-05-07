import cv2
import numpy as np
import mss
import pygetwindow as gw
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import signal
from collections import deque
from pynput.keyboard import Key, Controller

# ==============================================================================
# FILOSOFÍA DEL NUEVO ENFOQUE
# ==============================================================================
# ANTES: CNN ve 120x120 píxeles crudos → memoriza secuencias de imágenes del nivel
# AHORA: Extraemos un vector de estado semántico:
#   - ¿A qué distancia está el próximo obstáculo?
#   - ¿Cuál es su altura?
#   - ¿Está el cubo en el aire o en el suelo?
#   - ¿Cuál es la velocidad vertical del cubo? (diferencia entre frames)
#
# Esto permite generalizar a CUALQUIER nivel porque aprende la regla:
# "obstáculo cercano + cubo en suelo → saltar"
# en vez de "estos píxeles concretos → saltar"
# ==============================================================================

# ==============================================================================
# CONFIG
# ==============================================================================
GAME_TITLE      = "Geometry Dash"
TEMPLATE_PATH   = "attempt_template.png"
MODEL_SAVE_PATH = "gd_semantic_dqn.pth"
METRICS_PATH    = "training_log_v2.csv"
DEATH_THRESHOLD = 0.7
DEATH_COOLDOWN  = 0.5
DEBUG_VIEW      = True
SCALES          = np.linspace(0.4, 2.0, 20)
ROI_REL         = (0.3, 0.18, 0.7, 0.65)

# Parámetros de visión semántica
# El cubo siempre está en el tercio izquierdo de la pantalla
PLAYER_X_REL    = 0.25          # posición X relativa del cubo (25% de la pantalla)
SCAN_RANGE_REL  = (0.25, 0.85)  # rango horizontal donde buscar obstáculos
GROUND_Y_REL    = 0.78          # posición Y del suelo (relativa a la pantalla)
PLAYER_BAND_W   = 0.06          # ancho de banda para detectar cubo
N_SCAN_COLS     = 20            # columnas de escaneo para obstáculos
STATE_SIZE      = 8             # dimensión del vector de estado

# CONFIG ENTRENAMIENTO
BUFFER_SIZE     = 50_000
BATCH_SIZE      = 128
TRAIN_EVERY     = 2
TARGET_UPDATE   = 1000
GAMMA           = 0.99
LR              = 5e-4
EPSILON_START   = 1.0
EPSILON_END     = 0.05
EPSILON_DECAY   = 0.9995

# RECOMPENSAS
REWARD_ALIVE         = 0.1
REWARD_JUMP_PENALTY  = -0.05
REWARD_DEATH_BASE    = -10.0
REWARD_RECORD_BONUS  = 5.0
SURVIVAL_MILESTONES  = {3: 2.0, 6: 5.0, 10: 10.0, 15: 20.0, 20: 40.0}

# ==============================================================================
# RED NEURONAL: MLP pequeño (el estado semántico no necesita CNN)
# ==============================================================================
class SemanticDQN(nn.Module):
    """
    Red simple que recibe el vector de estado semántico.
    No necesita CNN porque ya recibe features extraídas, no píxeles.
    """
    def __init__(self, state_size=STATE_SIZE, history_len=4):
        super().__init__()
        input_size = state_size * history_len  # historial de estados para inferir velocidad/tendencia
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [no_jump, jump]
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# EXTRACTOR DE ESTADO SEMÁNTICO
# ==============================================================================
class SemanticStateExtractor:
    """
    Convierte un frame del juego en un vector de features semánticas:
    
    [0] dist_obstacle_1    - Distancia al obstáculo más cercano (normalizado 0-1)
    [1] height_obstacle_1  - Altura del obstáculo más cercano (normalizado 0-1)
    [2] dist_obstacle_2    - Distancia al segundo obstáculo (normalizado 0-1)
    [3] height_obstacle_2  - Altura del segundo obstáculo (normalizado 0-1)
    [4] player_y           - Posición Y del cubo (normalizado 0-1, 0=suelo)
    [5] player_airborne    - 1 si está en el aire, 0 si está en suelo
    [6] obstacle_density   - Densidad de obstáculos en zona próxima
    [7] gap_ahead          - 1 si hay un hueco/hoyo adelante, 0 si no
    """
    
    def __init__(self, win_width, win_height):
        self.w = win_width
        self.h = win_height
        self.ground_y = int(win_height * GROUND_Y_REL)
        self.player_x = int(win_width * PLAYER_X_REL)
        self.scan_x_start = int(win_width * SCAN_RANGE_REL[0])
        self.scan_x_end   = int(win_width * SCAN_RANGE_REL[1])
        self.prev_player_y = None
        
    def extract(self, frame_gray):
        """Extrae vector de estado semántico del frame gris."""
        h, w = frame_gray.shape
        
        # --- 1. Detectar obstáculos usando detección de bordes ---
        # Aplicar umbral adaptativo para encontrar objetos
        # Los obstáculos en GD son más oscuros/claros que el fondo
        blurred = cv2.GaussianBlur(frame_gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Zona de juego: excluir UI (arriba) y suelo
        play_zone_top    = int(h * 0.15)
        play_zone_bottom = int(h * 0.85)
        edges_cropped = edges[play_zone_top:play_zone_bottom, :]
        
        # --- 2. Encontrar posición del cubo (jugador) ---
        # El cubo está en una banda X fija, buscar su borde inferior
        band_w = int(w * PLAYER_BAND_W)
        player_band = edges_cropped[:, 
                                     max(0, self.player_x - band_w):
                                     min(w, self.player_x + band_w)]
        
        player_y_norm = 0.0  # 0 = suelo
        player_airborne = 0.0
        
        if player_band.any():
            rows_with_edges = np.where(player_band.any(axis=1))[0]
            if len(rows_with_edges) > 0:
                player_bottom_y = rows_with_edges[-1] + play_zone_top
                player_y_from_ground = self.ground_y - player_bottom_y
                player_y_norm = np.clip(player_y_from_ground / (h * 0.3), 0, 1)
                player_airborne = 1.0 if player_y_norm > 0.08 else 0.0
        
        # --- 3. Escanear columnas para detectar obstáculos ---
        scan_cols = np.linspace(self.scan_x_start, self.scan_x_end, 
                                 N_SCAN_COLS, dtype=int)
        
        obstacles = []
        for col_x in scan_cols:
            if col_x >= w: continue
            col = edges_cropped[:, col_x]
            edge_rows = np.where(col > 0)[0]
            if len(edge_rows) > 0:
                # El borde más bajo = tope del obstáculo
                obstacle_top = edge_rows[0] + play_zone_top
                obstacle_height = (self.ground_y - obstacle_top) / (h * 0.5)
                obstacle_height = np.clip(obstacle_height, 0, 1)
                
                dist_from_player = (col_x - self.player_x) / (w * 0.6)
                dist_norm = np.clip(dist_from_player, 0, 1)
                
                if dist_from_player > 0:  # solo obstáculos adelante
                    obstacles.append((dist_norm, obstacle_height, col_x))
        
        # Ordenar por distancia
        obstacles.sort(key=lambda x: x[0])
        
        # Features de obstáculos
        dist_1   = obstacles[0][0] if len(obstacles) > 0 else 1.0
        height_1 = obstacles[0][1] if len(obstacles) > 0 else 0.0
        dist_2   = obstacles[1][0] if len(obstacles) > 1 else 1.0
        height_2 = obstacles[1][1] if len(obstacles) > 1 else 0.0
        
        # Densidad de obstáculos en zona próxima (primero 30% del scan)
        close_threshold = 0.3
        density = len([o for o in obstacles if o[0] < close_threshold]) / max(N_SCAN_COLS * 0.3, 1)
        density = np.clip(density, 0, 1)
        
        # Detectar hoyo: si el suelo desaparece adelante
        ground_zone = frame_gray[self.ground_y:min(h, self.ground_y+15), 
                                  self.player_x:min(w, self.player_x + int(w*0.2))]
        gap_ahead = 0.0
        if ground_zone.size > 0:
            # Si el suelo es muy oscuro justo adelante, hay un hoyo
            mean_ground = np.mean(ground_zone)
            gap_ahead = 1.0 if mean_ground < 30 else 0.0
        
        state = np.array([
            dist_1, height_1,
            dist_2, height_2,
            player_y_norm, player_airborne,
            density, gap_ahead
        ], dtype=np.float32)
        
        return state, edges  # devuelve edges para debug


class StateHistory:
    """Mantiene historial de estados semánticos (equivalente al frame stack)."""
    def __init__(self, state_size=STATE_SIZE, history_len=4):
        self.history_len = history_len
        self.state_size  = state_size
        self.history = deque(maxlen=history_len)
    
    def reset(self):
        self.history.clear()
    
    def push(self, state_vec):
        if len(self.history) == 0:
            for _ in range(self.history_len):
                self.history.append(state_vec)
        else:
            self.history.append(state_vec)
    
    def get_tensor(self, device):
        flat = np.concatenate(list(self.history))
        return torch.FloatTensor(flat).unsqueeze(0).to(device)


# ==============================================================================
# REPLAY BUFFER CON PRIORIDAD PARA MUERTES
# ==============================================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer       = deque(maxlen=capacity)
        self.death_buffer = deque(maxlen=2000)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if done:
            self.death_buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 25% del batch son experiencias de muerte para reforzar señal negativa
        n_death  = min(int(batch_size * 0.25), len(self.death_buffer))
        n_normal = batch_size - n_death

        batch = random.sample(self.buffer, min(n_normal, len(self.buffer)))
        if n_death > 0:
            batch += random.sample(list(self.death_buffer), n_death)
        random.shuffle(batch)

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack([torch.FloatTensor(s) for s in states]),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack([torch.FloatTensor(s) for s in next_states]),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
def save_plots(history):
    if not history: return
    df = pd.DataFrame(history)
    window = 30
    
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
    plt.savefig('metrics_plot_v2.png', dpi=120)
    df.to_csv(METRICS_PATH, index=False)
    print(f"\n📊 Gráficas y CSV actualizados.")


def multiscale_match(gray, template, t_w, t_h, scales, threshold):
    best_score = -1
    best_loc, best_tw, best_th = None, t_w, t_h
    gh, gw_ = gray.shape[:2]
    for scale in scales:
        new_w, new_h = int(t_w * scale), int(t_h * scale)
        if new_w >= gw_ or new_h >= gh or new_w < 10 or new_h < 10: continue
        resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result  = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score, best_loc, best_tw, best_th = max_val, max_loc, new_w, new_h
    return (best_score, best_loc, 1.0, best_tw, best_th) if best_score >= threshold else None


def train_step(model, target_model, buffer, optimizer, device):
    if len(buffer) < max(BATCH_SIZE, 1000): return None  # warm-up mínimo
    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = (
        states.to(device), actions.to(device), rewards.to(device),
        next_states.to(device), dones.to(device)
    )
    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Double DQN: selecciona acción con model, evalúa con target_model
        best_actions  = model(next_states).argmax(1)
        next_q        = target_model(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
        target        = rewards + GAMMA * next_q * (1 - dones)

    loss = nn.SmoothL1Loss()(q_values, target)  # Huber loss, más estable que MSE
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
    optimizer.step()
    return loss.item()


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    running = True
    history = []

    def signal_handler(sig, frame):
        nonlocal running
        print("\n⏹ Guardando y cerrando...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Inicialización
    template_full = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if template_full is None:
        return print("❌ Error: No se encontró attempt_template.png")
    t_h, t_w = template_full.shape[:2]

    windows = gw.getWindowsWithTitle(GAME_TITLE)
    if not windows:
        return print("❌ Error: Juego no encontrado")
    win = windows[0]
    win.activate()

    monitor_full = {
        "top": win.top, "left": win.left,
        "width": win.width, "height": win.height
    }
    monitor_roi = {
        "left":   win.left   + int(win.width  * ROI_REL[0]),
        "top":    win.top    + int(win.height * ROI_REL[1]),
        "width":  int(win.width  * (ROI_REL[2] - ROI_REL[0])),
        "height": int(win.height * (ROI_REL[3] - ROI_REL[1]))
    }

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model        = SemanticDQN(STATE_SIZE, history_len=4).to(device)
    target_model = SemanticDQN(STATE_SIZE, history_len=4).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer    = optim.Adam(model.parameters(), lr=LR)
    scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    buffer       = ReplayBuffer(BUFFER_SIZE)
    keyboard     = Controller()
    extractor    = SemanticStateExtractor(win.width, win.height)
    state_hist   = StateHistory(STATE_SIZE, history_len=4)

    epsilon       = EPSILON_START
    episode       = 0
    best_time     = 0.0
    frame_count   = 0
    last_death_time  = 0
    is_dead          = False
    current_ep_reward   = 0.0
    episode_losses   = []
    milestones_reached = set()

    print(f"🧠 Entrenando en {device} con estado semántico.")
    print(f"   Estado: distancia/altura de obstáculos, posición cubo, densidad, hoyos")
    print(f"   Pulsa Ctrl+C para guardar y salir.\n")
    time.sleep(2)

    with mss.mss() as sct:
        # Estado inicial
        raw = np.array(sct.grab(monitor_full))
        gray_init = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        state_vec, _ = extractor.extract(gray_init)
        state_hist.push(state_vec)
        state_tensor = state_hist.get_tensor(device)
        attempt_start = time.perf_counter()

        while running:
            frame_count += 1
            now = time.perf_counter()

            # --- 1. Detección de Muerte ---
            roi_shot = sct.grab(monitor_roi)
            roi_gray = cv2.cvtColor(np.asarray(roi_shot), cv2.COLOR_BGRA2GRAY)
            match    = multiscale_match(roi_gray, template_full, t_w, t_h, SCALES, DEATH_THRESHOLD)

            if match is not None and not is_dead:
                is_dead          = True
                last_death_time  = now
                tiempo_vivo      = now - attempt_start
                episode         += 1
                milestones_reached = set()

                ratio        = min(tiempo_vivo / best_time, 1.0) if best_time > 0 else 0.0
                death_reward = REWARD_DEATH_BASE * (1.0 - ratio * 0.7)

                if tiempo_vivo > best_time:
                    death_reward += REWARD_RECORD_BONUS
                    best_time     = tiempo_vivo
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print(f"⭐ ¡Nuevo Récord! {best_time:.2f}s — Modelo guardado.")

                current_ep_reward += death_reward
                # Guardamos el estado actual como experiencia de muerte
                flat_state = state_tensor.squeeze(0).cpu().numpy()
                buffer.push(flat_state, 0, death_reward, flat_state, True)

                avg_loss = np.mean(episode_losses) if episode_losses else 0
                history.append({
                    'episode':    episode,
                    'time_alive': tiempo_vivo,
                    'reward':     current_ep_reward,
                    'loss':       avg_loss,
                    'epsilon':    epsilon
                })
                episode_losses    = []
                current_ep_reward = 0.0

                q_info = ""
                with torch.no_grad():
                    q = model(state_tensor)
                    q_info = f"Q=[{q[0][0]:.1f},{q[0][1]:.1f}]"
                print(f"💀 Ep {episode:4d} | t={tiempo_vivo:.2f}s | ε={epsilon:.3f} | {q_info}")

            elif is_dead and (now - last_death_time > DEATH_COOLDOWN):
                is_dead       = False
                attempt_start = time.perf_counter()
                state_hist.reset()
                raw = np.array(sct.grab(monitor_full))
                gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
                state_vec, _ = extractor.extract(gray)
                state_hist.push(state_vec)
                state_tensor = state_hist.get_tensor(device)

            # --- 2. IA y Acción ---
            if not is_dead:
                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

                if random.random() < epsilon:
                    action = random.randint(0, 1)
                else:
                    with torch.no_grad():
                        action = model(state_tensor).argmax().item()

                if action == 1:
                    keyboard.press(Key.space)
                    time.sleep(0.02)
                    keyboard.release(Key.space)

                # Siguiente estado
                raw_next  = np.array(sct.grab(monitor_full))
                gray_next = cv2.cvtColor(raw_next, cv2.COLOR_BGRA2GRAY)
                next_state_vec, debug_edges = extractor.extract(gray_next)
                state_hist.push(next_state_vec)
                next_state_tensor = state_hist.get_tensor(device)

                # Reward shaping con milestones
                tiempo_actual = now - attempt_start
                milestone_reward = 0.0
                for t_milestone, bonus in SURVIVAL_MILESTONES.items():
                    if tiempo_actual >= t_milestone and t_milestone not in milestones_reached:
                        milestones_reached.add(t_milestone)
                        milestone_reward += bonus
                        print(f"  🏁 Milestone {t_milestone}s alcanzado! +{bonus}")

                reward = REWARD_ALIVE + milestone_reward + (REWARD_JUMP_PENALTY if action == 1 else 0)
                current_ep_reward += reward

                flat_state      = state_tensor.squeeze(0).cpu().numpy()
                flat_next_state = next_state_tensor.squeeze(0).cpu().numpy()
                buffer.push(flat_state, action, reward, flat_next_state, False)
                state_tensor = next_state_tensor

                # Entrenar
                if frame_count % TRAIN_EVERY == 0:
                    l = train_step(model, target_model, buffer, optimizer, device)
                    if l:
                        episode_losses.append(l)
                        scheduler.step()

                if frame_count % TARGET_UPDATE == 0:
                    target_model.load_state_dict(model.state_dict())

                # Debug visual: mostrar edges detectados y estado semántico
                if DEBUG_VIEW:
                    debug_frame = cv2.cvtColor(gray_next, cv2.COLOR_GRAY2BGR)
                    # Dibujar línea del suelo
                    cv2.line(debug_frame, (0, extractor.ground_y),
                             (win.width, extractor.ground_y), (0, 255, 0), 1)
                    # Dibujar posición del cubo
                    cv2.line(debug_frame,
                             (extractor.player_x, 0),
                             (extractor.player_x, win.height), (255, 0, 0), 1)
                    # Mostrar estado semántico como texto
                    sv = next_state_vec
                    labels = [
                        f"dist1:{sv[0]:.2f} h1:{sv[1]:.2f}",
                        f"dist2:{sv[2]:.2f} h2:{sv[3]:.2f}",
                        f"py:{sv[4]:.2f} air:{sv[5]:.0f}",
                        f"dens:{sv[6]:.2f} gap:{sv[7]:.0f}",
                        f"e={epsilon:.3f} act={'JUMP' if action==1 else 'HOLD'}"
                    ]
                    for i, label in enumerate(labels):
                        cv2.putText(debug_frame, label, (5, 20 + i*18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    cv2.imshow("IA Vision - Semantic", debug_frame)
                    cv2.imshow("Edge Detection", debug_edges)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    # Finalización
    cv2.destroyAllWindows()
    torch.save(model.state_dict(), f"final_{MODEL_SAVE_PATH}")
    save_plots(history)
    print("✅ Guardado. Hasta pronto.")


if __name__ == "__main__":
    main()
