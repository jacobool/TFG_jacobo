import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

"""
gd_rl_maml_cubo.py

Meta-learning estilo Reptile (FOMAML) sobre PPO para Geometry Dash.

Idea
----
Fase 1  (meta-train, sobre cubo 1):
    Definimos una *distribucion de tareas* perturbando los pesos de la
    recompensa. Para cada tarea hacemos un inner-loop corto con PPO y
    aplicamos el meta-update de Reptile:
        theta_meta <- theta_meta + epsilon * (theta_adaptado - theta_meta)
    Asi obtenemos un theta* que, en lugar de ser optimo para una sola
    configuracion, esta preparado para adaptarse rapido a cualquiera.

Fase 2  (few-shot, sobre cubo 2):
    Cargamos theta*, ejecutamos pocos miles de pasos de PPO sobre la
    segunda fase del nivel y comparamos contra un baseline entrenado
    desde cero con los mismos pasos.

Limitaciones honestas
---------------------
- Reptile != MAML. Aproximamos el meta-gradiente con un primer orden;
  evitamos los Hessianos pero perdemos el doble paso del inner-update.
  En tareas estrechas como esta, la diferencia practica es pequena.
- La distribucion de tareas es estrecha: solo perturbamos pesos de
  reward sobre el mismo nivel. Lo ideal seria *muchos* niveles
  distintos, pero eso requiere paralelizar Geometry Dash, cosa que
  el setup real-time no permite.
- Usamos PPO porque MAML-RL no encaja con DQN/QR-DQN (off-policy +
  replay buffer rompen la formulacion).

Uso paso a paso
---------------
1) Coloca Geometry Dash con el cubo 1 visible y la ventana enfocada.
2) Lanza meta-train (tarda; ~20 iter * ~2 min = 40 min reales):
       python gd_rl_maml_cubo.py --mode meta_train
3) Cambia el nivel a cubo 2 (segunda fase, start position 2 manual).
4) Lanza few-shot:
       python gd_rl_maml_cubo.py --mode few_shot \
           --base modelos_guardados/maml_meta_FINAL.zip
5) Evalua el resultado con jugar_gd_4.py apuntando al .zip generado.

Comparativa para la memoria
---------------------------
- baseline:       PPO scratch en cubo 2 con args.adapt_steps timesteps.
- maml:           PPO con theta* de meta-train + args.adapt_steps en cubo 2.
- diferencia:     metrica clave = pasos hasta superar X% del cubo 2.
"""

import argparse
import csv
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


class EpisodeCSVLogger(BaseCallback):
    """Escribe (episodio, timestep_relativo, reward, longitud) a CSV.

    Usa un contador LOCAL (_local_step) en vez de self.num_timesteps,
    porque al reanudar un modelo (p.ej. few-shot tras meta-train) el
    contador global arrastra los pasos del meta-train y aplasta la
    curva en la grafica.
    """
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.ep_counter = 0
        self._local_step = 0
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["episode", "timestep", "reward", "length"])
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

from gd_rl_env_4 import GDEnv as GDEnvBase


# ---------------------------------------------------------------------------
# 1. TASKWRAPPER: perturba la recompensa para construir una mini-distribucion
# ---------------------------------------------------------------------------
class TaskWrapper(gym.Wrapper):
    """
    Multiplica el reward por un factor escalar muestreado de una lista de
    presets. Cada preset = una 'tarea' diferente desde la perspectiva del
    meta-learner. Se mantiene la dinamica del nivel; solo cambian los
    incentivos.
    """

    # (factor_global, sesgo_supervivencia)
    # factor_global escala todo el reward; sesgo_supervivencia anade un
    # bonus por step vivo. Asi cubrimos un abanico desde 'corre rapido'
    # (factor alto, bonus bajo) hasta 'sobrevive a toda costa' (factor
    # bajo, bonus alto).
    REWARD_PRESETS = [
        (1.00, 0.00),   # canonica
        (1.20, 0.05),   # mas premio + bonus de supervivencia
        (0.80, 0.10),   # menos premio + bonus alto
        (1.40, 0.00),   # solo amplifica
        (0.60, 0.20),   # cauteloso
    ]

    def __init__(self, env, task_id=0):
        super().__init__(env)
        self.task_id = task_id
        self._apply(task_id)

    def _apply(self, task_id):
        self.task_id = task_id
        self.factor, self.alive_bonus = self.REWARD_PRESETS[task_id]

    def set_task(self, task_id):
        self._apply(task_id)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = reward * self.factor
        if not (terminated or truncated):
            shaped += self.alive_bonus
        info = dict(info)
        info["task_id"] = self.task_id
        return obs, shaped, terminated, truncated, info


# ---------------------------------------------------------------------------
# 2. UTILIDADES PARA SNAPSHOT / RESTORE / META-UPDATE
# ---------------------------------------------------------------------------
def snapshot(model):
    return {k: v.detach().clone() for k, v in model.policy.state_dict().items()}

def restore(model, snap):
    model.policy.load_state_dict(snap, strict=True)

def reptile_step(model, theta_pre, theta_post, eps):
    """theta_meta <- theta_pre + eps * (theta_post - theta_pre)"""
    nuevo = {k: theta_pre[k] + eps * (theta_post[k] - theta_pre[k])
             for k in theta_pre}
    model.policy.load_state_dict(nuevo, strict=True)


# ---------------------------------------------------------------------------
# 3. ENV BUILDER
# ---------------------------------------------------------------------------
def build_env(task_id=0):
    def _thunk():
        return TaskWrapper(GDEnvBase(), task_id=task_id)
    venv = DummyVecEnv([_thunk])
    venv = VecFrameStack(venv, n_stack=4)
    return venv


def get_taskwrapper(venv):
    """Recupera el TaskWrapper interno tras envolver con DummyVecEnv+stack."""
    inner = venv.venv.envs[0] if hasattr(venv, "venv") else venv.envs[0]
    return inner


# ---------------------------------------------------------------------------
# 4. META-TRAIN (Reptile)
# ---------------------------------------------------------------------------
def meta_train(args):
    Path("modelos_guardados").mkdir(exist_ok=True)
    print(f"[MAML/Reptile] iters={args.meta_iters}  "
          f"inner_steps={args.inner_steps}  meta_lr={args.meta_lr}")

    env = build_env(task_id=0)
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        device="auto",
    )
    print(f"[MAML/Reptile] PPO inicializado, device={model.device}")

    n_tasks = len(TaskWrapper.REWARD_PRESETS)
    eps = args.meta_lr
    rng = random.Random(args.seed)
    log = []

    for it in range(args.meta_iters):
        task_id = rng.randrange(n_tasks)
        get_taskwrapper(env).set_task(task_id)
        factor, bonus = TaskWrapper.REWARD_PRESETS[task_id]
        print(f"\n[meta-iter {it+1}/{args.meta_iters}] "
              f"tarea={task_id} (factor={factor}, bonus={bonus}) "
              f"eps={eps:.3f}")

        theta_pre = snapshot(model)
        t0 = time.perf_counter()
        model.learn(total_timesteps=args.inner_steps,
                    reset_num_timesteps=False, progress_bar=False)
        dt = time.perf_counter() - t0

        theta_post = snapshot(model)
        reptile_step(model, theta_pre, theta_post, eps)

        eps = max(0.05, eps * args.meta_decay)
        log.append((it + 1, task_id, dt))
        print(f"  inner {args.inner_steps} pasos en {dt:.1f}s; "
              f"meta-update aplicado")

        if (it + 1) % args.ckpt_every == 0:
            p = f"modelos_guardados/maml_meta_iter{it+1}.zip"
            model.save(p)
            print(f"  [ckpt] {p}")

    final = "modelos_guardados/maml_meta_FINAL.zip"
    model.save(final)
    print(f"\n[MAML/Reptile] listo -> {final}")
    print(f"  iteraciones={len(log)}  "
          f"tiempo_total={sum(d for _,_,d in log):.0f}s")


# ---------------------------------------------------------------------------
# 5. FEW-SHOT ADAPTATION sobre cubo 2
# ---------------------------------------------------------------------------
def few_shot(args):
    print(f"[MAML/Reptile] few-shot  base={args.base}  "
          f"adapt_steps={args.adapt_steps}")
    print("  Asegurate de tener Geometry Dash en la fase 2 del cubo y "
          "la ventana enfocada antes de continuar.")
    input("  Pulsa ENTER cuando este listo... ")

    env = build_env(task_id=0)  # canonica: queremos optimizar el reward real
    model = PPO.load(args.base, env=env)
    print(f"  modelo cargado, device={model.device}")

    csv_path = f"metrics/maml_fewshot_{args.out_suffix}.csv"
    cb = EpisodeCSVLogger(csv_path)
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.adapt_steps,
                reset_num_timesteps=False, progress_bar=False, callback=cb)
    dt = time.perf_counter() - t0

    out = (f"modelos_guardados/maml_{args.out_suffix}_adapted_"
           f"{args.adapt_steps}.zip")
    model.save(out)
    print(f"[MAML/Reptile] adaptado -> {out}  ({dt:.0f}s)  "
          f"log -> {csv_path}")


# ---------------------------------------------------------------------------
# 6. BASELINE: PPO scratch sobre cubo 2 con los mismos pasos (para comparar)
# ---------------------------------------------------------------------------
def baseline_scratch(args):
    print(f"[BASELINE] PPO scratch sobre cubo 2  steps={args.adapt_steps}")
    print("  Asegurate de tener cubo 2 visible y la ventana enfocada.")
    input("  Pulsa ENTER cuando este listo... ")

    env = build_env(task_id=0)
    model = PPO("CnnPolicy", env, learning_rate=2.5e-4, n_steps=256,
                batch_size=64, n_epochs=4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, verbose=0, device="auto")
    csv_path = f"metrics/baseline_scratch_{args.out_suffix}.csv"
    cb = EpisodeCSVLogger(csv_path)
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.adapt_steps,
                reset_num_timesteps=False, progress_bar=False, callback=cb)
    dt = time.perf_counter() - t0

    out = (f"modelos_guardados/ppo_scratch_{args.out_suffix}_"
           f"{args.adapt_steps}.zip")
    model.save(out)
    print(f"[BASELINE] -> {out}  ({dt:.0f}s)  "
          f"log -> {csv_path}")


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["meta_train", "few_shot", "baseline_scratch"])
    ap.add_argument("--meta_iters", type=int, default=20,
                    help="numero de iteraciones Reptile (cada una = 1 tarea)")
    ap.add_argument("--inner_steps", type=int, default=2000,
                    help="timesteps de PPO por tarea en el inner-loop")
    ap.add_argument("--meta_lr", type=float, default=0.4,
                    help="epsilon de Reptile (paso del meta-update)")
    ap.add_argument("--meta_decay", type=float, default=0.97,
                    help="decay multiplicativo de meta_lr por iter")
    ap.add_argument("--ckpt_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=str,
                    default="modelos_guardados/maml_meta_FINAL.zip")
    ap.add_argument("--adapt_steps", type=int, default=8000,
                    help="timesteps para few-shot / baseline scratch")
    ap.add_argument("--out_suffix", type=str, default="cubo2",
                    help="sufijo para CSV y .zip (p.ej. cubo2, nivel2)")
    args = ap.parse_args()

    if args.mode == "meta_train":
        meta_train(args)
    elif args.mode == "few_shot":
        few_shot(args)
    else:
        baseline_scratch(args)
