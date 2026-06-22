import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Importamos del entorno v2 (con buffer retroactivo)
from gd_rl_nave_2 import GDEnv, STEP_DURATION, detect_player_and_band, get_window_rect

# --- CONFIGURACIÓN ---
CHECKPOINT = "modelos_guardados/nave_dqn_1_800000_steps"
N_EPISODIOS = 10
EDGE_TOP_THRESHOLD = 0.12
EDGE_BOT_THRESHOLD = 0.88


# --- ENTORNO DIAGNÓSTICO ---
class GDEnvDiag(GDEnv):
    """
    Extiende GDEnv (v2, con buffer retroactivo) para registrar frame a frame
    el estado del agente durante la evaluación.

    Problema resuelto respecto a la versión anterior:
    Cuando done=True, la pantalla ya muestra la reaparición del jugador,
    por lo que recapturar el frame daría una y_norm incorrecta (la del respawn).
    Se guarda '_diag_last_alive_y': la última posición Y conocida mientras
    el jugador era visible. En el frame de muerte se usa ese valor guardado,
    que corresponde al estado real de la colisión.
    """

    def reset(self, seed=None, options=None):
        # Guardamos el log del episodio anterior ANTES de que DummyVecEnv
        # lo limpie con el reset automático al recibir done=True
        self.last_diag_log = getattr(self, 'diag_log', []).copy()
        self.diag_log = []
        self._diag_last_alive_y = 0.5  # Última Y válida (jugador visible)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)

        # Recapturamos pantalla para diagnóstico
        import mss as mss_lib
        monitor = get_window_rect(self.hwnd)
        with mss_lib.mss() as sct:
            img = np.array(sct.grab(monitor))

        frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h = frame_bgr.shape[0]
        player_cy, green_area, _ = detect_player_and_band(frame_bgr)
        y_norm_actual = float(np.clip(player_cy / max(1, h), 0.0, 1.0))

        # Si el jugador es visible, actualizamos la última Y conocida
        if green_area > 0:
            self._diag_last_alive_y = y_norm_actual

        # En el frame de muerte, la pantalla puede mostrar la reaparición
        # (y_norm incorrecto). Usamos la última posición válida en su lugar.
        y_norm_log = self._diag_last_alive_y if done else y_norm_actual

        edge_triggered = (y_norm_log < EDGE_TOP_THRESHOLD or y_norm_log > EDGE_BOT_THRESHOLD)

        self.diag_log.append({
            "frame":        len(self.diag_log),
            "y_norm":       y_norm_log,
            "action":       int(info.get("action_applied", action)),
            "green_area":   float(green_area),
            "edge_penalty": edge_triggered,
            "death":        bool(done),
            "reward":       float(reward),
        })

        return obs, reward, done, truncated, info


# --- VISUALIZACIÓN DE UN EPISODIO ---
def plot_episodio(log, ep_num):
    df = pd.DataFrame(log)
    if df.empty:
        return

    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    frames = df["frame"]

    # --- 1. Posición Y ---
    ax = axs[0]
    ax.plot(frames, df["y_norm"], color="steelblue", linewidth=1.2, label="Posición Y")
    ax.axhline(EDGE_TOP_THRESHOLD, color="red",    linestyle="--", alpha=0.7,
               label=f"Límite techo ({EDGE_TOP_THRESHOLD})")
    ax.axhline(EDGE_BOT_THRESHOLD, color="orange", linestyle="--", alpha=0.7,
               label=f"Límite suelo ({EDGE_BOT_THRESHOLD})")
    ax.axhspan(0.0, EDGE_TOP_THRESHOLD, alpha=0.08, color="red")
    ax.axhspan(EDGE_BOT_THRESHOLD, 1.0, alpha=0.08, color="orange")

    edge_frames = df[df["edge_penalty"]]
    ax.scatter(edge_frames["frame"], edge_frames["y_norm"],
               color="red", s=15, zorder=5, label="Edge penalty activado")

    death_frames = df[df["death"]]
    if not death_frames.empty:
        ax.axvline(death_frames["frame"].iloc[-1], color="black",
                   linestyle="-", linewidth=2, label="MUERTE")

    ax.set_ylabel("Y relativa (0=techo, 1=suelo)")
    ax.set_ylim(-0.05, 1.05)
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"Episodio {ep_num} — Posición vertical del jugador")
    ax.grid(alpha=0.2)

    # --- 2. Acción tomada ---
    ax2 = axs[1]
    ax2.fill_between(frames, df["action"], step="post",
                     color="mediumseagreen", alpha=0.7, label="Acción (1=pulsar)")
    ax2.set_ylabel("Acción")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Soltar (0)", "Pulsar (1)"])
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    for i in range(len(df) - 1):
        color = "lightgreen" if df["action"].iloc[i] == 1 else "lightyellow"
        ax2.axvspan(df["frame"].iloc[i], df["frame"].iloc[i + 1], alpha=0.3, color=color)

    # --- 3. Recompensa acumulada ---
    ax3 = axs[2]
    cum_reward = df["reward"].cumsum()
    ax3.plot(frames, cum_reward, color="darkgreen", linewidth=1.2)
    ax3.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax3.set_ylabel("Reward acumulado")
    ax3.set_xlabel("Frame")
    ax3.grid(alpha=0.2)

    plt.suptitle(f"Diagnóstico Episodio {ep_num} — Checkpoint 800k", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"diag_ep{ep_num:02d}.png", dpi=110)
    plt.close()
    print(f"  📊 Guardado: diag_ep{ep_num:02d}.png")


# --- ANÁLISIS GLOBAL ---
def analisis_global(todos_los_logs):
    if not todos_los_logs:
        print("❌ No hay logs para analizar.")
        return

    todos = pd.concat([pd.DataFrame(log) for log in todos_los_logs], ignore_index=True)
    total_frames = len(todos)

    frames_peligro_techo = (todos["y_norm"] < EDGE_TOP_THRESHOLD).sum()
    frames_peligro_suelo = (todos["y_norm"] > EDGE_BOT_THRESHOLD).sum()
    frames_edge = todos["edge_penalty"].sum()

    print("\n" + "=" * 55)
    print("  ANÁLISIS GLOBAL DE COMPORTAMIENTO")
    print("=" * 55)
    print(f"  Total frames registrados : {total_frames}")
    print(f"  Frames en zona techo     : {frames_peligro_techo} "
          f"({100 * frames_peligro_techo / total_frames:.1f}%)")
    print(f"  Frames en zona suelo     : {frames_peligro_suelo} "
          f"({100 * frames_peligro_suelo / total_frames:.1f}%)")
    print(f"  Total edge penalties     : {int(frames_edge)} "
          f"({100 * frames_edge / total_frames:.1f}% del tiempo)")

    VENTANA_PRE_MUERTE = 10
    muertes_con_edge = 0
    muertes_en_techo = 0
    n_eps = len(todos_los_logs)

    for log in todos_los_logs:
        df = pd.DataFrame(log)
        if df.empty or not df["death"].any():
            continue
        muerte_idx = df[df["death"]].index[0]
        ventana = df.iloc[max(0, muerte_idx - VENTANA_PRE_MUERTE): muerte_idx + 1]
        if ventana["edge_penalty"].any():
            muertes_con_edge += 1
        y_muerte = df.loc[muerte_idx, "y_norm"]
        if y_muerte < EDGE_TOP_THRESHOLD + 0.10:
            muertes_en_techo += 1

    print(f"\n  Muertes precedidas de edge penalty (ventana {VENTANA_PRE_MUERTE}f):")
    print(f"    {muertes_con_edge}/{n_eps} episodios ({100 * muertes_con_edge / max(1, n_eps):.0f}%)")
    print(f"  Muertes cerca del techo  : {muertes_en_techo}/{n_eps} "
          f"({100 * muertes_en_techo / max(1, n_eps):.0f}%)")

    if muertes_en_techo / max(1, n_eps) > 0.5:
        print("\n  ⚠️  DIAGNÓSTICO: El agente muere principalmente por el TECHO.")
        print("      El edge penalty no está corrigiendo este comportamiento.")
        print("      → Threshold demasiado estricto o penalización demasiado pequeña.")
    else:
        print("\n  ✅ Las muertes están distribuidas. El techo no es el problema principal.")

    # Histograma de Y en frame de muerte
    y_muertes_flat = []
    for log in todos_los_logs:
        df = pd.DataFrame(log)
        muertes = df[df["death"]]
        if not muertes.empty:
            y_muertes_flat.append(muertes["y_norm"].values[0])

    if y_muertes_flat:
        plt.figure(figsize=(8, 4))
        plt.hist(y_muertes_flat, bins=20, color="steelblue", edgecolor="white")
        plt.axvline(EDGE_TOP_THRESHOLD, color="red",    linestyle="--", label="Límite techo")
        plt.axvline(EDGE_BOT_THRESHOLD, color="orange", linestyle="--", label="Límite suelo")
        plt.xlabel("Posición Y relativa en frame de muerte (0=techo, 1=suelo)")
        plt.ylabel("Número de muertes")
        plt.title("¿Dónde muere el agente?")
        plt.legend()
        plt.tight_layout()
        plt.savefig("diag_donde_muere.png", dpi=110)
        plt.close()
        print("\n  📊 Guardado: diag_donde_muere.png")

    todos.to_csv("diag_todos_frames.csv", index=False)
    print("  💾 Guardado: diag_todos_frames.csv")


# --- MAIN ---
if __name__ == "__main__":
    env = DummyVecEnv([lambda: GDEnvDiag()])
    env = VecFrameStack(env, n_stack=4)
    model = DQN.load(CHECKPOINT, env=env)

    print(f"🔍 Diagnóstico con checkpoint 800k — {N_EPISODIOS} episodios")
    print("⚠️  Deja el juego en primer plano.\n")

    todos_los_logs = []
    obs = env.reset()

    for ep in range(1, N_EPISODIOS + 1):
        done = False
        inner_env = env.venv.envs[0]

        while not done:
            step_start = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            elapsed = time.perf_counter() - step_start
            if STEP_DURATION - elapsed > 0:
                time.sleep(STEP_DURATION - elapsed)

        # last_diag_log fue guardado por reset() antes de que DummyVecEnv
        # limpiara el episodio con el reset automático
        log = inner_env.last_diag_log

        if not log:
            print(f"  Ep {ep:02d}: log vacío, episodio demasiado corto")
            obs = env.reset()
            continue

        frames_techo = sum(1 for f in log if f["y_norm"] < EDGE_TOP_THRESHOLD)
        print(f"  Ep {ep:02d}: {len(log)} frames | "
              f"Techo: {frames_techo} frames ({100 * frames_techo / max(1, len(log)):.0f}%) | "
              f"Edge penalties: {sum(1 for f in log if f['edge_penalty'])}")

        plot_episodio(log, ep)
        todos_los_logs.append(log)
        obs = env.reset()

    analisis_global(todos_los_logs)
    env.close()