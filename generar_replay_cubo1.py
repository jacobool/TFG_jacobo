"""Genera un replay buffer del cubo 1 reutilizando el modelo ya entrenado.

Carga gd_qrdqn_440000_steps.zip (sin tocar pesos), juega N pasos en el
cubo 1 con politica epsilon-greedy y guarda las transiciones como pickle
para que despues el script de adaptacion al cubo 2 las inyecte en el
buffer y evite el olvido catastrofico (CLEAR / experience replay mixing).

ANTES DE EJECUTAR:
  - Pon Geometry Dash en primer plano con el nivel del cubo 1 abierto
    (Start Position al PRINCIPIO del nivel, no en el cubo 2).
  - El modelo se ejecuta en modo determinista + epsilon = 0.03 para que
    haya algo de diversidad en las transiciones recogidas.
  - Cuando el script termine, encontraras models/replay_cubo1.pkl listo
    para usar en gd_rl_env_4_qrdqn_replay_cubo2.py.
"""

import os
import time
import numpy as np
from sb3_contrib import QRDQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from gd_rl_env_4 import GDEnv


# ---------------------------------------------------------------- Config
SOURCE_CHECKPOINT = "modelos_guardados/gd_qrdqn_440000_steps.zip"
OUTPUT_BUFFER     = "models/replay_cubo1.pkl"

# Cuantas transiciones recolectar. 30k es suficiente para tener variedad
# del cubo 1 sin que la sesion dure mas de ~50 minutos de juego.
N_TRANSICIONES = 30_000

# Capacidad del buffer al guardarlo. Tiene que ser >= la capacidad del
# buffer del modelo destino (cubo 2) para que load_replay_buffer no
# trunque. Ponemos 200k para que quepan las del cubo 1 + las del cubo 2.
BUFFER_CAPACITY = 200_000

# Epsilon para la politica de exploracion (NO afecta a los pesos).
EPSILON = 0.03


# ----------------------------------------------------------- Main
def main():
    os.makedirs(os.path.dirname(OUTPUT_BUFFER), exist_ok=True)

    print("== Generando replay buffer del cubo 1 ==")
    print(f"   Modelo origen : {SOURCE_CHECKPOINT}")
    print(f"   Salida        : {OUTPUT_BUFFER}")
    print(f"   Transiciones  : {N_TRANSICIONES:,}")
    print(f"   Epsilon       : {EPSILON}")
    print()
    print("Pon Geometry Dash en cubo 1 (inicio del nivel) y NO toques.")
    print("Empieza la captura en 5 segundos...")
    time.sleep(5)

    # 1) Entorno identico al de entrenamiento (frame stack incluido).
    env = DummyVecEnv([lambda: GDEnv()])
    env = VecFrameStack(env, n_stack=4)

    # 2) Carga del modelo. NO se entrena: solo se usa para predecir.
    # SB3 envuelve internamente con VecTransposeImage para pasar de HWC
    # (84,84,4) a CHW (4,84,84): hay que usar SU env (model.env) y SU
    # observation_space al construir el buffer, no los locales, para que
    # el buffer quede en CHW y QRDQN pueda anhadir transiciones despues
    # sin error de broadcast.
    print(f"Cargando {SOURCE_CHECKPOINT}...")
    model = QRDQN.load(SOURCE_CHECKPOINT, env=env, device="auto")
    model.policy.set_training_mode(False)
    env = model.env  # ya envuelto con VecTransposeImage

    # 3) Buffer destino en CHW. Mismo formato que usa SB3 internamente,
    # asi load_replay_buffer lo aceptara sin conversiones.
    buffer = ReplayBuffer(
        buffer_size=BUFFER_CAPACITY,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device="cpu",
        n_envs=1,
        optimize_memory_usage=False,
    )

    # 4) Bucle de recoleccion.
    obs = env.reset()
    n_eps = 0
    t0 = time.time()

    for step in range(N_TRANSICIONES):
        # Politica epsilon-greedy a partir del modelo.
        if np.random.rand() < EPSILON:
            action = np.array([env.action_space.sample()])
        else:
            action, _ = model.predict(obs, deterministic=True)

        new_obs, reward, done, info = env.step(action)

        # SB3 ReplayBuffer.add() espera arrays con dimension de envs.
        buffer.add(
            obs=obs,
            next_obs=new_obs,
            action=action,
            reward=reward,
            done=done,
            infos=info,
        )

        obs = new_obs
        if done.any():
            n_eps += 1
            obs = env.reset()

        if (step + 1) % 1000 == 0:
            elapsed = time.time() - t0
            tps = (step + 1) / elapsed
            eta_s = (N_TRANSICIONES - step - 1) / max(tps, 1e-6)
            print(f"  step {step+1:>6}/{N_TRANSICIONES} | "
                  f"eps={n_eps} | {tps:.1f} tps | ETA {eta_s/60:.1f} min")

    # 5) Persistencia.
    print()
    print(f"Guardando buffer en {OUTPUT_BUFFER}...")
    # Truco: la API publica de SB3 expone save_replay_buffer en los
    # modelos, no en el buffer; asignamos temporalmente al modelo y
    # llamamos a su metodo.
    model.replay_buffer = buffer
    model.save_replay_buffer(OUTPUT_BUFFER)

    size_mb = os.path.getsize(OUTPUT_BUFFER) / 1024 / 1024
    print(f"Listo. {N_TRANSICIONES:,} transiciones | "
          f"{n_eps} episodios | {size_mb:.1f} MB en disco.")
    env.close()


if __name__ == "__main__":
    main()
