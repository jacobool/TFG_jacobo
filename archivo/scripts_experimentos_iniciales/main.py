import cv2
import numpy as np
import mss
import pygetwindow as gw
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pynput.keyboard import Key, Controller

# ==============================================================================
# CONFIG GENERAL
# ==============================================================================

GAME_TITLE       = "Geometry Dash"       # o "Geometry Dash Lite"
TEMPLATE_PATH    = "attempt_template.png"
THRESHOLD        = 0.7
DEATH_COOLDOWN   = 0.5
DEBUG_VIEW       = True
SCALES           = np.linspace(0.4, 2.0, 20)
ROI_REL          = (0.3, 0.18, 0.7, 0.65)
SCREEN_SIZE      = (120, 120)

# ==============================================================================
# CONFIG ENTRENAMIENTO
# ==============================================================================

BUFFER_SIZE      = 10_000   # Tamaño del replay buffer
BATCH_SIZE       = 64       # Muestras por paso de entrenamiento
TRAIN_EVERY      = 4        # Entrenar cada N frames
TARGET_UPDATE    = 500      # Actualizar target network cada N frames
GAMMA            = 0.99     # Factor de descuento
LR               = 1e-4     # Learning rate
EPSILON_START    = 1.0      # Exploración inicial (100%)
EPSILON_END      = 0.05     # Exploración mínima (5%)
EPSILON_DECAY    = 0.9995   # Decaimiento por frame

# ==============================================================================
# CONFIG RECOMPENSAS
# ==============================================================================

REWARD_ALIVE          = 1.0    # Recompensa por cada frame vivo
REWARD_JUMP_PENALTY   = -0.05  # Penalización por saltar (evitar spam)
REWARD_DEATH_BASE     = -80.0 # Penalización base por morir
REWARD_RECORD_BONUS   = 70.0   # Bonus por superar el récord personal
# Penalización escalonada: morir muy pronto vs tu récord penaliza más
# Fórmula: death_penalty = REWARD_DEATH_BASE * (1 - ratio_progreso * 0.8)

# ==============================================================================
# ARQUITECTURA CNN (DQN)
# ==============================================================================

class GeometryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 11 * 11, 256),
            nn.ReLU(),
            nn.Linear(256, 2)   # 0 = no saltar, 1 = saltar
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ==============================================================================
# REPLAY BUFFER
# ==============================================================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.cat(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.cat(next_states),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

# ==============================================================================
# DETECCIÓN DE MUERTE - Template matching multi-escala
# ==============================================================================

def compute_roi_monitor(win, roi_rel):
    x0, y0, x1, y1 = roi_rel
    return {
        "left":   win.left + int(win.width  * x0),
        "top":    win.top  + int(win.height * y0),
        "width":  int(win.width  * (x1 - x0)),
        "height": int(win.height * (y1 - y0))
    }


def multiscale_match(gray, template, t_w, t_h, scales, threshold):
    best_score = -1
    best_loc   = None
    best_scale = 1.0
    best_tw, best_th = t_w, t_h
    gh, gw_ = gray.shape[:2]

    for scale in scales:
        new_w = int(t_w * scale)
        new_h = int(t_h * scale)
        if new_w >= gw_ or new_h >= gh or new_w < 10 or new_h < 10:
            continue
        resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result  = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
            best_loc   = max_loc
            best_scale = scale
            best_tw, best_th = new_w, new_h

    if best_score >= threshold:
        return best_score, best_loc, best_scale, best_tw, best_th
    return None

# ==============================================================================
# CAPTURA DE PANTALLA
# ==============================================================================

def get_screen(sct, monitor, device):
    img     = np.array(sct.grab(monitor))
    gray    = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    resized = cv2.resize(gray, SCREEN_SIZE)
    tensor  = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0).to(device) / 255.0
    return resized, tensor

# ==============================================================================
# PASO DE ENTRENAMIENTO ONLINE (DQN con target network)
# ==============================================================================

def train_step(model, target_model, buffer, optimizer, device):
    if len(buffer) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
    states      = states.to(device)
    actions     = actions.to(device)
    rewards     = rewards.to(device)
    next_states = next_states.to(device)
    dones       = dones.to(device)

    # Q(s, a) actual
    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Q target: r + gamma * max Q'(s', a') si no terminó
    with torch.no_grad():
        next_q = target_model(next_states).max(1)[0]
        target = rewards + GAMMA * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # --- Template ---
    template_full = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if template_full is None:
        print(f"ERROR: No se pudo cargar '{TEMPLATE_PATH}'"); return
    t_h, t_w = template_full.shape[:2]
    print(f"Template cargado: {t_w}x{t_h} px")

    # --- Ventana ---
    windows = gw.getWindowsWithTitle(GAME_TITLE)
    if not windows:
        print(f"No se encontró la ventana '{GAME_TITLE}'."); return
    win = windows[0]
    try:
        win.activate()
    except Exception as e:
        print(f"Aviso activate(): {e} — continuando igualmente")
    print(f"Ventana: ({win.left},{win.                   top}) {win.width}x{win.height}")

    monitor_full = {"top": win.top, "left": win.left,
                    "width": win.width, "height": win.height}
    monitor_roi  = compute_roi_monitor(win, ROI_REL)

    # --- Modelo ---
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model        = GeometryCNN().to(device)
    target_model = GeometryCNN().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer    = optim.Adam(model.parameters(), lr=LR)
    buffer       = ReplayBuffer(BUFFER_SIZE)
    keyboard     = Controller()

    # --- Estado ---
    epsilon         = EPSILON_START
    frame_count     = 0
    episode         = 0
    best_time       = 0.0          # récord personal en segundos
    attempt_start   = time.perf_counter()
    is_dead         = False
    last_death_time = 0.0
    first_attempt   = True         # el primer intento no cuenta para el récord

    time.sleep(2)
    print("Iniciando IA online con recompensa por tiempo vivo...")
    print(f"Dispositivo: {device}")

    with mss.mss() as sct:
        _, state_tensor = get_screen(sct, monitor_full, device)

        while True:
            frame_count += 1
            now = time.perf_counter()

            # ------------------------------------------------------------------
            # 1. DETECCIÓN DE MUERTE
            # ------------------------------------------------------------------
            roi_shot = sct.grab(monitor_roi)
            roi_gray = cv2.cvtColor(np.asarray(roi_shot), cv2.COLOR_BGRA2GRAY)
            match    = multiscale_match(roi_gray, template_full, t_w, t_h,
                                        SCALES, THRESHOLD)

            death_detected_this_frame = False

            if match is not None:
                score, loc, scale, tw, th = match
                if not is_dead:
                    is_dead                   = True
                    last_death_time           = now
                    death_detected_this_frame = True
                    tiempo_vivo               = now - attempt_start
                    episode += 1

                    if first_attempt:
                        # El primer intento no cuenta (timing impreciso)
                        first_attempt = False
                        print(f"⏭️  Ep {episode:4d} | "
                              f"t={tiempo_vivo:.2f}s | "
                              f"PRIMER INTENTO DESCARTADO")
                    else:
                        nueva_best = tiempo_vivo > best_time

                        # Recompensa de muerte proporcional al progreso vs récord
                        ratio        = min(tiempo_vivo / best_time, 1.0) if best_time > 0 else 0.0
                        death_reward = REWARD_DEATH_BASE * (1.0 - ratio * 0.8)
                        if nueva_best:
                            death_reward += REWARD_RECORD_BONUS
                            best_time = tiempo_vivo

                        print(f"💀 Ep {episode:4d} | "
                              f"t={tiempo_vivo:.2f}s | "
                              f"record={best_time:.2f}s | "
                              f"r_muerte={death_reward:.1f} | "
                              f"ε={epsilon:.3f}")

                        buffer.push(state_tensor, 0, death_reward, state_tensor, True)

            else:
                # Recuperación tras muerte
                if is_dead and (now - last_death_time > DEATH_COOLDOWN):
                    is_dead       = False
                    attempt_start = time.perf_counter()

            # ------------------------------------------------------------------
            # 2. DEBUG VIEW
            # ------------------------------------------------------------------
            if DEBUG_VIEW:
                debug = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
                if match is not None:
                    score, loc, scale, tw, th = match
                    cv2.rectangle(debug, loc, (loc[0]+tw, loc[1]+th), (0,255,0), 2)
                    cv2.putText(debug, f"score={score:.2f}",
                                (loc[0], max(loc[1]-8, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                status = "MUERTO" if is_dead else "VIVO"
                color  = (0,0,255) if is_dead else (0,255,0)
                t_vivo = now - attempt_start if not is_dead else 0.0
                cv2.putText(debug,
                            f"[{status}] t={t_vivo:.1f}s rec={best_time:.1f}s ep={episode}",
                            (5, debug.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
                cv2.imshow("Debug - Deteccion Muerte", debug)

            # ------------------------------------------------------------------
            # 3. LÓGICA DE IA (solo cuando está vivo)
            # ------------------------------------------------------------------
            if is_dead:
                time.sleep(1.0)
                _, state_tensor = get_screen(sct, monitor_full, device)
                cv2.imshow("IA Vision", np.zeros(SCREEN_SIZE, dtype=np.uint8))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Epsilon-greedy con decaimiento por frame
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

            next_img, next_state_tensor = get_screen(sct, monitor_full, device)

            # Recompensa por frame vivo
            reward = REWARD_ALIVE
            if action == 1:
                reward += REWARD_JUMP_PENALTY

            if not death_detected_this_frame:
                buffer.push(state_tensor, action, reward, next_state_tensor, False)

            state_tensor = next_state_tensor

            # ------------------------------------------------------------------
            # 4. ENTRENAMIENTO ONLINE
            # ------------------------------------------------------------------
            if frame_count % TRAIN_EVERY == 0:
                train_step(model, target_model, buffer, optimizer, device)

            if frame_count % TARGET_UPDATE == 0:
                target_model.load_state_dict(model.state_dict())

            cv2.imshow("IA Vision", next_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print(f"\nEntrenamiento finalizado. Mejor tiempo: {best_time:.2f}s | Episodios: {episode}")


if __name__ == "__main__":
    main()