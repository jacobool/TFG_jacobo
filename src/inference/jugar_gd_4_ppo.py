import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

import sys
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from jugar_gd_4 import GDEnv

DEFAULT_MODEL = "gd_ppo_FINAL_4"


def main(model_path):
    print(f"🤖 Cargando modelo PPO desde '{model_path}'...")

    env = DummyVecEnv([lambda: GDEnv()])
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load(model_path, env=env)
    print(f"✅ Modelo cargado.")

    print("\n🚀 Evaluación iniciada. Ctrl+C para parar.")
    print("   Asegúrate de tener Geometry Dash en primer plano.")

    try:
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)
            if dones[0]:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Evaluación detenida.")
    finally:
        env.close()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    main(path)
