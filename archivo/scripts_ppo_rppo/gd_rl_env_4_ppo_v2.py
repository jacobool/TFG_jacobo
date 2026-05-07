import os
import time
import numpy as np
import pydirectinput
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from gd_rl_env_4 import GDEnv, detect_player_and_band, GREEN_AREA_MIN, DEATH_FRAMES_NEEDED, STEP_DURATION
from gd_rl_env_4_ppo import MetricsAndSaveCallback, save_plots, PPO_LOG_KEYS

RUN_NAME = "ppo_v2"
METRICS_DIR = "metrics"
SAVE_FREQ = 10000
TOTAL_TIMESTEPS = 500000

# Penalización por salto suavizada (era -0.04). Mantiene el incentivo a no saltar
# de forma gratuita, pero evita el mínimo local "nunca saltar" que estanca a PPO.
JUMP_PENALTY = -0.01


class GDEnvPPOv2(GDEnv):
    """Mismo entorno que GDEnv pero con penalización por salto más suave."""

    def step(self, action):
        step_start = time.perf_counter()
        reward = 0.05
        done = False

        now = time.perf_counter()
        if action == 1 and (now - self.last_action_time > 0.05):
            pydirectinput.keyDown("space")
            pydirectinput.keyUp("space")
            reward += JUMP_PENALTY
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
        remaining = STEP_DURATION - elapsed
        if remaining > 0:
            time.sleep(remaining)

        return obs, reward, done, False, {"attempt": self.attempt, "green": green_area}


class VecNormalizeSavingCallback(MetricsAndSaveCallback):
    """Extiende el callback base para guardar también los stats de VecNormalize."""

    def _on_step(self) -> bool:
        proceed = super()._on_step()
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            vn_path = f"{self.save_path}_{self.num_timesteps}_steps_vecnormalize.pkl"
            self.training_env.save(vn_path)
        return proceed


if __name__ == "__main__":
    os.makedirs("modelos_guardados", exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    env = DummyVecEnv([lambda: GDEnvPPOv2()])
    env = VecFrameStack(env, n_stack=4)
    # Normalización del reward: reduce el impacto del -5 de muerte sobre el value head.
    # clip_reward=10 evita outliers tras normalizar. Las obs (uint8 imagen) NO se normalizan.
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)

    callback = VecNormalizeSavingCallback(
        save_freq=SAVE_FREQ,
        save_path=f"modelos_guardados/gd_{RUN_NAME}",
        run_name=RUN_NAME,
        log_keys=PPO_LOG_KEYS,
    )

    # PPO v2 — ajustes para salir del óptimo local "no saltar nunca":
    #   n_steps=512     → rollouts de ~34s, ≥6 episodios por update, GAE más estable
    #   n_epochs=10     → más reúso de cada rollout (compensa ser on-policy)
    #   batch_size=128  → mini-batches mayores, gradiente menos ruidoso
    #   ent_coef=0.001  → 10× menos presión de entropía (ya no colapsa la señal de gradiente)
    #   learning_rate=1e-4 → más conservador para single-env con reward ruidoso
    #   net_arch separado: value head con más capacidad para que explained_variance suba
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        learning_rate=1e-4,
        clip_range=0.1,
        ent_coef=0.001,
        vf_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128], vf=[256, 256])),
    )

    print(f"🚀 Entrenamiento PPO v2 — {TOTAL_TIMESTEPS} pasos, guardado cada {SAVE_FREQ}.")
    print(f"   JUMP_PENALTY={JUMP_PENALTY} (era -0.04) | VecNormalize reward activo")
    print(f"   Métricas → {METRICS_DIR}/{RUN_NAME}_metrics.csv, {METRICS_DIR}/{RUN_NAME}_plot.png")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    model.save(f"gd_{RUN_NAME}_FINAL_4")
    env.save(f"gd_{RUN_NAME}_FINAL_4_vecnormalize.pkl")
    save_plots(callback.history, RUN_NAME, show=True)
    env.close()
    print("✅ PPO v2 terminado.")
