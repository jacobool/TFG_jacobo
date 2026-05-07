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
import sys
from collections import deque
from pynput.keyboard import Key, Controller

# ==============================================================================
# CONFIG GENERAL Y PERSISTENCIA
# ==============================================================================
GAME_TITLE      = "Geometry Dash"
TEMPLATE_PATH   = "attempt_template.png"
MODEL_SAVE_PATH = "geometry_dash_dqn.pth"
METRICS_PATH    = "training_log.csv"
THRESHOLD       = 0.7
DEATH_COOLDOWN  = 0.5
DEBUG_VIEW      = True
SCALES          = np.linspace(0.4, 2.0, 20)
ROI_REL         = (0.3, 0.18, 0.7, 0.65)
SCREEN_SIZE     = (120, 120)

# CONFIG ENTRENAMIENTO
BUFFER_SIZE     = 10_000
BATCH_SIZE      = 64
TRAIN_EVERY     = 4
TARGET_UPDATE   = 800
GAMMA           = 0.99
LR              = 1e-4
EPSILON_START   = 1.0
EPSILON_END     = 0.05
EPSILON_DECAY   = 0.999
FRAME_STACK_K   = 4

# RECOMPENSAS
REWARD_ALIVE        = 1.0
REWARD_JUMP_PENALTY = -0.25
REWARD_DEATH_BASE   = -50.0
REWARD_RECORD_BONUS = 25.0

# ==============================================================================
# ARQUITECTURA Y CLASES APOYO
# ==============================================================================
class GeometryCNN4F(nn.Module):
    def __init__(self, frame_stack=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 11 * 11, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class FrameStack:
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self):
        self.frames.clear()

    def push(self, frame_tensor):
        if len(self.frames) == 0:
            for _ in range(self.k): self.frames.append(frame_tensor)
        else:
            self.frames.append(frame_tensor)

    def get_state(self):
        return torch.cat(list(self.frames), dim=1)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.cat(states), torch.tensor(actions), 
                torch.tensor(rewards, dtype=torch.float32), 
                torch.cat(next_states), torch.tensor(dones, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
def save_plots(history):
    if not history: return
    df = pd.DataFrame(history)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    axs[0].plot(df['episode'], df['time_alive'], color='blue')
    axs[0].set_title('Tiempo de Supervivencia por Episodio')
    axs[0].set_ylabel('Segundos')

    axs[1].plot(df['episode'], df['reward'], color='green')
    axs[1].set_title('Recompensa Total')
    axs[1].set_ylabel('Reward')

    axs[2].plot(df['episode'], df['loss'], color='red')
    axs[2].set_title('Pérdida (Loss) Promedio')
    axs[2].set_ylabel('MSE Loss')

    plt.tight_layout()
    plt.savefig('metrics_plot.png')
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
        result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score, best_loc, best_tw, best_th = max_val, max_loc, new_w, new_h
    return (best_score, best_loc, 1.0, best_tw, best_th) if best_score >= threshold else None

def get_screen(sct, monitor, device):
    img = np.array(sct.grab(monitor))
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    resized = cv2.resize(gray, SCREEN_SIZE)
    tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0).to(device) / 255.0
    return resized, tensor

def train_step(model, target_model, buffer, optimizer, device):
    if len(buffer) < BATCH_SIZE: return None
    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = states.to(device), actions.to(device), rewards.to(device), next_states.to(device), dones.to(device)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_model(next_states).max(1)[0]
        target = rewards + GAMMA * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, target)
    optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10); optimizer.step()
    return loss.item()

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # Variables de control para el cierre
    running = True
    history = []

    def signal_handler(sig, frame):
        nonlocal running
        print("\nArrancando procedimiento de guardado y cierre...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Inicialización
    template_full = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if template_full is None: return print("Error: No template")
    t_h, t_w = template_full.shape[:2]

    windows = gw.getWindowsWithTitle(GAME_TITLE)
    if not windows: return print("Error: Game not found")
    win = windows[0]; win.activate()

    monitor_full = {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
    monitor_roi  = {"left": win.left + int(win.width * ROI_REL[0]), "top": win.top + int(win.height * ROI_REL[1]), 
                    "width": int(win.width * (ROI_REL[2]-ROI_REL[0])), "height": int(win.height * (ROI_REL[3]-ROI_REL[1]))}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeometryCNN4F().to(device)
    target_model = GeometryCNN4F().to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)
    keyboard = Controller()
    frame_stack = FrameStack(k=FRAME_STACK_K)

    epsilon, episode, best_time = EPSILON_START, 0, 0.0
    frame_count, last_death_time, is_dead = 0, 0, False
    current_episode_reward = 0
    episode_losses = []

    print(f"Entrenando en {device}. Pulsa Ctrl+C para salir y guardar.")
    time.sleep(2)

    with mss.mss() as sct:
        _, f_tensor = get_screen(sct, monitor_full, device)
        frame_stack.push(f_tensor)
        state_tensor = frame_stack.get_state()
        attempt_start = time.perf_counter()

        while running:
            frame_count += 1
            now = time.perf_counter()

            # 1. Detección de Muerte
            roi_shot = sct.grab(monitor_roi)
            roi_gray = cv2.cvtColor(np.asarray(roi_shot), cv2.COLOR_BGRA2GRAY)
            match = multiscale_match(roi_gray, template_full, t_w, t_h, SCALES, THRESHOLD)

            if match is not None and not is_dead:
                is_dead, last_death_time = True, now
                tiempo_vivo = now - attempt_start
                episode += 1
                
                # Calcular recompensa de muerte
                ratio = min(tiempo_vivo / best_time, 1.0) if best_time > 0 else 0.0
                death_reward = REWARD_DEATH_BASE * (1.0 - ratio * 0.8)
                
                if tiempo_vivo > best_time:
                    death_reward += REWARD_RECORD_BONUS
                    best_time = tiempo_vivo
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print(f"⭐ ¡Nuevo Récord! Modelo guardado.")

                current_episode_reward += death_reward
                buffer.push(state_tensor, 0, death_reward, state_tensor, True)
                
                # Registrar métricas del episodio
                avg_loss = np.mean(episode_losses) if episode_losses else 0
                history.append({
                    'episode': episode, 'time_alive': tiempo_vivo, 
                    'reward': current_episode_reward, 'loss': avg_loss
                })
                episode_losses, current_episode_reward = [], 0
                
                print(f"💀 Ep {episode} | t={tiempo_vivo:.2f}s | r_total={history[-1]['reward']:.1f}")

            elif is_dead and (now - last_death_time > DEATH_COOLDOWN):
                is_dead, attempt_start = False, time.perf_counter()
                frame_stack.reset()
                _, f_tensor = get_screen(sct, monitor_full, device)
                frame_stack.push(f_tensor)
                state_tensor = frame_stack.get_state()

            # 2. IA y Acción
            if not is_dead:
                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
                if random.random() < epsilon: action = random.randint(0, 1)
                else:
                    with torch.no_grad(): action = model(state_tensor).argmax().item()

                if action == 1:
                    keyboard.press(Key.space); time.sleep(0.02); keyboard.release(Key.space)

                next_img, next_f_tensor = get_screen(sct, monitor_full, device)
                frame_stack.push(next_f_tensor)
                next_state_tensor = frame_stack.get_state()

                reward = REWARD_ALIVE + (REWARD_JUMP_PENALTY if action == 1 else 0)
                current_episode_reward += reward
                buffer.push(state_tensor, action, reward, next_state_tensor, False)
                state_tensor = next_state_tensor

                if frame_count % TRAIN_EVERY == 0:
                    l = train_step(model, target_model, buffer, optimizer, device)
                    if l: episode_losses.append(l)
                if frame_count % TARGET_UPDATE == 0:
                    target_model.load_state_dict(model.state_dict())
                # Trackea el epsilon y Q-values medios
                if frame_count % 100 == 0:
                    with torch.no_grad():
                        q_vals = model(state_tensor)
                        print(f"ε={epsilon:.3f} | Q[no_jump]={q_vals[0][0]:.2f} | Q[jump]={q_vals[0][1]:.2f}")
                if DEBUG_VIEW:
                    cv2.imshow("IA Vision", next_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Finalización
    cv2.destroyAllWindows()
    torch.save(model.state_dict(), f"final_{MODEL_SAVE_PATH}")
    save_plots(history)
    print("Saliendo de forma segura.")

if __name__ == "__main__":
    main()