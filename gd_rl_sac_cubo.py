"""
gd_rl_sac_cubo.py

Soft Actor-Critic (SAC) sobre Geometry Dash (cubo).

Por que SAC en un problema discreto?
------------------------------------
SAC nativo (SB3) solo soporta espacios de accion *continuos* (Box).
Nuestro GDEnv tiene Discrete(2) (saltar / no saltar). Para que SAC se
pueda ejecutar sobre el mismo entorno, envolvemos el entorno con un
wrapper que:

    - Expone Box(low=-1, high=+1, shape=(1,)) al algoritmo.
    - Internamente convierte la accion continua a discreta:
          a_discreta = 1 si a_continua[0] >= 0 else 0

De este modo la politica de SAC aprende a producir un escalar y
nosotros lo binarizamos antes de aplicarlo al juego. No es la
formulacion mas eficiente para un problema binario (DQN/QR-DQN encajan
mejor), pero permite tener una *baseline off-policy con politica
estocastica continua* para comparar.

Salida:
    - modelos_guardados/sac_cubo1_FINAL.zip
    - metrics/sac_cubo1_metrics.csv (curvas por episodio)

Uso:
    python gd_rl_sac_cubo.py --timesteps 200000
"""

import argparse
import csv
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from gd_rl_env_4 import GDEnv as GDEnvBase


# ---------------------------------------------------------------------------
# 1. WRAPPER: Box continuo -> Discrete(2)
# ---------------------------------------------------------------------------
class ContinuousToDiscreteAction(gym.Wrapper):
    """
    Expone al agente un Box([-1], [+1]) y lo convierte a Discrete(2)
    antes de pasarlo al entorno real. Threshold en 0:
        a >= 0  -> saltar (1)
        a < 0   -> no saltar (0)
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(1,), dtype=np.float32)

    def step(self, action):
        a_disc = int(action[0] >= 0)
        return self.env.step(a_disc)


# ---------------------------------------------------------------------------
# 2. CSV LOGGER
# ---------------------------------------------------------------------------
class EpisodeCSVLogger(BaseCallback):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["episode", "timestep", "reward", "length"])
        self.ep_counter = 0
        self._local_step = 0
        self._ep_reward = 0.0
        self._ep_length = 0

    def _on_step(self):
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])
        self._ep_reward += float(rewards[0])
        self._ep_length += 1
        self._local_step += 1
        if bool(dones[0]):
            self.ep_counter += 1
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([self.ep_counter, self._local_step,
                                        self._ep_reward, self._ep_length])
            self._ep_reward = 0.0
            self._ep_length = 0
        return True


# ---------------------------------------------------------------------------
# 3. ENV BUILDER
# ---------------------------------------------------------------------------
def build_env():
    def _thunk():
        return ContinuousToDiscreteAction(GDEnvBase())
    venv = DummyVecEnv([_thunk])
    venv = VecFrameStack(venv, n_stack=4)
    return venv


# ---------------------------------------------------------------------------
# 4. ENTRENAMIENTO
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--out", default="modelos_guardados/sac_cubo1_FINAL.zip")
    ap.add_argument("--csv", default="metrics/sac_cubo1_metrics.csv")
    args = ap.parse_args()

    Path("modelos_guardados").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    print(f"[SAC] timesteps={args.timesteps}  out={args.out}")
    print("  Asegurate de tener Geometry Dash con cubo 1 visible y la "
          "ventana enfocada antes de continuar.")
    input("  Pulsa ENTER cuando este listo... ")

    env = build_env()
    model = SAC(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        learning_starts=1000,
        ent_coef="auto",
        verbose=0,
        device="auto",
    )
    print(f"[SAC] inicializado, device={model.device}")

    cb = EpisodeCSVLogger(args.csv)
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps,
                reset_num_timesteps=True, callback=cb, progress_bar=False)
    dt = time.perf_counter() - t0

    model.save(args.out)
    print(f"[SAC] guardado -> {args.out}  ({dt:.0f}s)")
    print(f"[SAC] curvas   -> {args.csv}")


if __name__ == "__main__":
    main()
