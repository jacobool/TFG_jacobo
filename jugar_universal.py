"""
jugar_universal.py - Evalua cualquier modelo entrenado (DQN/QR-DQN/PPO/
RecurrentPPO/SAC/PEARL) detectando el tipo automaticamente por el nombre.

Uso:
    python jugar_universal.py --model modelos_guardados/gd_qrdqn_440000_steps
    python jugar_universal.py --model modelos_guardados/sac_cubo1_FINAL.zip
    python jugar_universal.py --model modelos_guardados/maml_cubo2_adapted_8000.zip
    python jugar_universal.py --model modelos_guardados/pearl_FINAL.pt

Reglas de deteccion (por sustring en el nombre del fichero):
    sac          -> SAC      (necesita wrapper continuo->discreto)
    recurrentppo -> RecurrentPPO
    qrdqn        -> QR-DQN
    ppo / maml   -> PPO      (maml_*_adapted_* es un modelo PPO)
    dqn          -> DQN
    *.pt         -> PEARL    (PyTorch puro, loop custom)
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from jugar_gd_4 import GDEnv  # entorno ligero ya existente


# ---------------------------------------------------------------------------
# 1. Wrapper SAC/PEARL: Box([-1,+1]) -> Discrete(2)
# ---------------------------------------------------------------------------
import gymnasium as gym
from gymnasium import spaces


class ContinuousToDiscreteAction(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(1,), dtype=np.float32)

    def step(self, action):
        return self.env.step(int(action[0] >= 0))


# ---------------------------------------------------------------------------
# 2. Deteccion del algoritmo
# ---------------------------------------------------------------------------
def detect_algo(path: str) -> str:
    p = Path(path).name.lower()
    if p.endswith(".pt"):
        return "pearl"
    n = p.replace(".zip", "")
    if "sac" in n:
        return "sac"
    if "recurrentppo" in n:
        return "recurrentppo"
    if "qrdqn" in n:
        return "qrdqn"
    if "ppo" in n or "maml" in n:
        return "ppo"
    if "dqn" in n:
        return "dqn"
    raise ValueError(f"No detecto el algoritmo en '{p}'. "
                     f"Pasa --algo manualmente.")


# ---------------------------------------------------------------------------
# 3. Inferencia para modelos SB3 (DQN, QR-DQN, PPO, SAC, RecurrentPPO)
# ---------------------------------------------------------------------------
def play_sb3(model_path: str, algo: str):
    # Wrapper extra para SAC
    if algo == "sac":
        env = DummyVecEnv([lambda: ContinuousToDiscreteAction(GDEnv())])
    else:
        env = DummyVecEnv([lambda: GDEnv()])
    env = VecFrameStack(env, n_stack=4)

    if algo == "qrdqn":
        from sb3_contrib import QRDQN
        AlgoCls = QRDQN
    elif algo == "recurrentppo":
        from sb3_contrib import RecurrentPPO
        AlgoCls = RecurrentPPO
    elif algo == "sac":
        AlgoCls = SAC
    elif algo == "ppo":
        AlgoCls = PPO
    else:
        AlgoCls = DQN

    model = AlgoCls.load(model_path, env=env)
    print(f"[OK] '{model_path}' cargado como {AlgoCls.__name__}.")
    print("Ctrl+C para detener.\n")

    try:
        obs = env.reset()
        # RecurrentPPO necesita pasar el estado LSTM
        if algo == "recurrentppo":
            lstm_states = None
            ep_start = np.ones((1,), dtype=bool)
            while True:
                action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=ep_start,
                    deterministic=True)
                obs, _, dones, _ = env.step(action)
                ep_start = dones
        else:
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _ = env.step(action)
    except KeyboardInterrupt:
        print("\n[stop] interrumpido por el usuario.")
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 4. Inferencia para PEARL (PyTorch puro, single env, sin VecFrameStack)
# ---------------------------------------------------------------------------
def play_pearl(model_path: str):
    import torch as th
    from gd_rl_pearl_cubo import PEARL, LATENT_DIM, DEVICE

    agent = PEARL(n_tasks=1)
    agent.load(model_path)
    env = ContinuousToDiscreteAction(GDEnv())
    print(f"[OK] '{model_path}' cargado como PEARL (device={DEVICE}).")
    print("Ctrl+C para detener.\n")

    # Sin contexto en inferencia: usamos z ~ prior N(0, I).
    # Es lo que hace PEARL para la primera trayectoria de una tarea nueva.
    z = th.zeros(1, LATENT_DIM, device=DEVICE)

    try:
        obs, _ = env.reset()
        while True:
            a = agent.act(obs, z)
            obs, _, term, trunc, _ = env.step(np.array([float(a[0])]))
            if term or trunc:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\n[stop] interrumpido por el usuario.")
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="ruta al .zip (SB3) o .pt (PEARL)")
    ap.add_argument("--algo", default=None,
                    help="forzar tipo (dqn/qrdqn/ppo/sac/recurrentppo/pearl)")
    args = ap.parse_args()

    algo = args.algo or detect_algo(args.model)
    print(f"[detectado] {args.model} -> {algo.upper()}")

    if algo == "pearl":
        play_pearl(args.model)
    else:
        play_sb3(args.model, algo)
