import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

"""
gd_rl_pearl_cubo.py

Implementacion adaptada (PEARL-lite) de:
  Rakelly et al., "Efficient Off-Policy Meta-RL via Probabilistic Context
  Variables", ICML 2019.

Ideas que conserva
------------------
- Backbone off-policy estilo SAC.
- Variable latente probabilistica z ~ q(z|c) inferida desde un *contexto*
  c = lista de transiciones (s, a, r, s') de la tarea actual.
- Politica y critic condicionados en (estado, z).
- Posterior sampling: distintos z generan exploraciones distintas en la
  misma tarea.
- Entrenamiento sobre una distribucion de tareas. Aqui, las "tareas"
  son las mismas perturbaciones de reward que en gd_rl_maml_cubo.py
  (REWARD_PRESETS), para que la comparacion entre meta-algoritmos sea
  consistente.

Simplificaciones honestas (vs PEARL original)
---------------------------------------------
1. Encoder: PEARL original procesa el contexto con una red factorizada
   producto-de-Gaussianas. Aqui usamos un MLP que produce mu y log_var
   directamente desde una concatenacion de transiciones agrupadas.
2. Sin entropia automatica de SAC: alpha es un hiperparametro fijo.
3. Observaciones: aplicamos un CNN feature extractor pequeno antes de
   concatenar con z, en lugar de las redes totalmente conectadas del
   paper (que asume estados de baja dimension).
4. Accion continua [-1, +1] -> binarizada por threshold (mismo wrapper
   que gd_rl_sac_cubo.py).
5. Per-task replay buffers reducidos: 5000 transiciones por tarea, no
   millones, por restricciones de RAM y tiempo del entorno real-time.

Uso
---
    python gd_rl_pearl_cubo.py --mode meta_train --timesteps 60000
    python gd_rl_pearl_cubo.py --mode adapt --base modelos_guardados/pearl_FINAL.pt

Salida:
    modelos_guardados/pearl_FINAL.pt
    metrics/pearl_metrics.csv
"""

import argparse
import csv
import random
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from gd_rl_env_4 import GDEnv as GDEnvBase
from gd_rl_maml_cubo import TaskWrapper


DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
LATENT_DIM = 8
CONTEXT_LEN = 16   # numero de transiciones que ve el encoder
HIDDEN = 256


# ---------------------------------------------------------------------------
# 1. WRAPPER continuo -> discreto (igual que en SAC)
# ---------------------------------------------------------------------------
class ContinuousToDiscreteAction(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(1,), dtype=np.float32)

    def step(self, action):
        a_disc = int(action[0] >= 0)
        return self.env.step(a_disc)


def make_task_env(task_id):
    base = GDEnvBase()
    return ContinuousToDiscreteAction(TaskWrapper(base, task_id=task_id))


# ---------------------------------------------------------------------------
# 2. REDES
# ---------------------------------------------------------------------------
class CNNExtractor(nn.Module):
    """CNN minimo que reduce (84,84,1) a un vector de 128 floats."""
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, out_dim), nn.ReLU(),
        )

    def forward(self, obs):  # obs: (B, 84, 84, 1) float32 [0,1]
        x = obs.permute(0, 3, 1, 2)
        return self.net(x)


class ContextEncoder(nn.Module):
    """Encoder q(z|c). Recibe un batch de transiciones y produce mu, logvar."""
    def __init__(self, feat_dim, action_dim=1, latent_dim=LATENT_DIM):
        super().__init__()
        in_dim = feat_dim + action_dim + 1 + feat_dim  # phi(s)+a+r+phi(s')
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.mu_head = nn.Linear(HIDDEN, latent_dim)
        self.logvar_head = nn.Linear(HIDDEN, latent_dim)

    def forward(self, ctx):  # ctx: (B, K, in_dim)
        x = self.net(ctx)
        x = x.mean(dim=1)  # agregamos por media (en vez de prod-de-gauss)
        return self.mu_head(x), self.logvar_head(x).clamp(-10, 2)


class Actor(nn.Module):
    def __init__(self, feat_dim, latent_dim=LATENT_DIM, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + latent_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.mu_head = nn.Linear(HIDDEN, action_dim)
        self.log_std_head = nn.Linear(HIDDEN, action_dim)

    def forward(self, feat, z):
        x = self.net(th.cat([feat, z], dim=-1))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(-5, 2)
        return mu, log_std

    def sample(self, feat, z):
        mu, log_std = self.forward(feat, z)
        std = log_std.exp()
        eps = th.randn_like(mu)
        a = th.tanh(mu + std * eps)  # squash a [-1, 1]
        # log-prob con correccion de tanh
        log_p = (-0.5 * ((eps) ** 2) - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        log_p -= th.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, log_p


class Critic(nn.Module):
    def __init__(self, feat_dim, latent_dim=LATENT_DIM, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + action_dim + latent_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, feat, a, z):
        return self.net(th.cat([feat, a, z], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# 3. REPLAY BUFFER POR TAREA
# ---------------------------------------------------------------------------
class TaskBuffer:
    def __init__(self, capacity=5000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s.astype(np.uint8), float(a), float(r),
                          s2.astype(np.uint8), float(done)))

    def sample(self, n):
        idx = np.random.randint(0, len(self.buf), size=n)
        batch = [self.buf[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        return (np.stack(s), np.array(a, dtype=np.float32),
                np.array(r, dtype=np.float32), np.stack(s2),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# 4. AGENTE PEARL
# ---------------------------------------------------------------------------
class PEARL:
    def __init__(self, n_tasks, lr=3e-4, gamma=0.99, alpha=0.2, kl_w=0.1):
        self.n_tasks = n_tasks
        self.gamma, self.alpha, self.kl_w = gamma, alpha, kl_w
        self.feat = CNNExtractor().to(DEVICE)
        self.encoder = ContextEncoder(128).to(DEVICE)
        self.actor = Actor(128).to(DEVICE)
        self.q1 = Critic(128).to(DEVICE)
        self.q2 = Critic(128).to(DEVICE)
        self.q1_t = Critic(128).to(DEVICE); self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t = Critic(128).to(DEVICE); self.q2_t.load_state_dict(self.q2.state_dict())
        self.opt_actor = th.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_q = th.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.opt_enc = th.optim.Adam(
            list(self.encoder.parameters()) + list(self.feat.parameters()), lr=lr)

    def _to_obs(self, x):
        x = th.as_tensor(x, dtype=th.float32, device=DEVICE) / 255.0
        return x

    def _build_context(self, buffer, k=CONTEXT_LEN):
        if len(buffer) < k:
            return None
        s, a, r, s2, _ = buffer.sample(k)
        with th.no_grad():
            f = self.feat(self._to_obs(s))
            f2 = self.feat(self._to_obs(s2))
        a_t = th.as_tensor(a, device=DEVICE).unsqueeze(-1)
        r_t = th.as_tensor(r, device=DEVICE).unsqueeze(-1)
        return th.cat([f, a_t, r_t, f2], dim=-1).unsqueeze(0)  # (1, K, dim)

    def sample_z(self, ctx):
        if ctx is None:
            return th.zeros(1, LATENT_DIM, device=DEVICE), 0.0
        mu, logvar = self.encoder(ctx)
        std = (0.5 * logvar).exp()
        eps = th.randn_like(mu)
        z = mu + eps * std
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
        return z, kl

    def act(self, obs, z, deterministic=False):
        with th.no_grad():
            f = self.feat(self._to_obs(obs[None]))
            mu, log_std = self.actor.forward(f, z)
            if deterministic:
                a = th.tanh(mu)
            else:
                a, _ = self.actor.sample(f, z)
        return a.cpu().numpy()[0]

    def update(self, buffers, batch_size=64):
        # Muestreamos una tarea, obtenemos contexto y batch RL
        task_id = random.randrange(self.n_tasks)
        buf = buffers[task_id]
        if len(buf) < CONTEXT_LEN + batch_size:
            return None
        ctx = self._build_context(buf)
        z, kl = self.sample_z(ctx)
        z_b = z.expand(batch_size, -1)

        s, a, r, s2, d = buf.sample(batch_size)
        s = self._to_obs(s); s2 = self._to_obs(s2)
        a = th.as_tensor(a, device=DEVICE).unsqueeze(-1)
        r = th.as_tensor(r, device=DEVICE)
        d = th.as_tensor(d, device=DEVICE)

        f = self.feat(s); f2 = self.feat(s2)

        # Critic
        with th.no_grad():
            a2, logp2 = self.actor.sample(f2, z_b)
            q1_t = self.q1_t(f2, a2, z_b)
            q2_t = self.q2_t(f2, a2, z_b)
            q_t = th.min(q1_t, q2_t) - self.alpha * logp2
            target = r + (1 - d) * self.gamma * q_t
        q1 = self.q1(f, a, z_b); q2 = self.q2(f, a, z_b)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.opt_q.zero_grad(); self.opt_enc.zero_grad()
        (q_loss + self.kl_w * kl).backward(retain_graph=True)
        self.opt_q.step(); self.opt_enc.step()

        # Actor
        a_pi, logp = self.actor.sample(f.detach(), z_b.detach())
        q_pi = th.min(self.q1(f.detach(), a_pi, z_b.detach()),
                      self.q2(f.detach(), a_pi, z_b.detach()))
        actor_loss = (self.alpha * logp - q_pi).mean()
        self.opt_actor.zero_grad(); actor_loss.backward(); self.opt_actor.step()

        # Soft update
        for tgt, src in [(self.q1_t, self.q1), (self.q2_t, self.q2)]:
            for tp, sp in zip(tgt.parameters(), src.parameters()):
                tp.data.mul_(0.995).add_(0.005 * sp.data)

        return {"q_loss": q_loss.item(), "actor_loss": actor_loss.item(),
                "kl": float(kl) if isinstance(kl, th.Tensor) else 0.0,
                "task": task_id}

    def save(self, path):
        th.save({"feat": self.feat.state_dict(),
                 "encoder": self.encoder.state_dict(),
                 "actor": self.actor.state_dict(),
                 "q1": self.q1.state_dict(), "q2": self.q2.state_dict()}, path)

    def load(self, path):
        s = th.load(path, map_location=DEVICE)
        self.feat.load_state_dict(s["feat"])
        self.encoder.load_state_dict(s["encoder"])
        self.actor.load_state_dict(s["actor"])
        self.q1.load_state_dict(s["q1"]); self.q2.load_state_dict(s["q2"])


# ---------------------------------------------------------------------------
# 5. META-TRAIN
# ---------------------------------------------------------------------------
def meta_train(args):
    Path("modelos_guardados").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)
    n_tasks = len(TaskWrapper.REWARD_PRESETS)

    print(f"[PEARL] meta-train  timesteps={args.timesteps}  tareas={n_tasks}")
    print("  Asegurate de tener Geometry Dash con cubo 1 visible y la "
          "ventana enfocada.")
    input("  Pulsa ENTER cuando este listo... ")

    agent = PEARL(n_tasks=n_tasks)
    buffers = [TaskBuffer() for _ in range(n_tasks)]

    csv_path = "metrics/pearl_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "timestep", "task", "reward", "length"])
    ep = 0; total_steps = 0
    t0 = time.perf_counter()

    # Como abrir 5 ventanas de GD no es viable, alternamos la *misma*
    # ventana cambiando solo la tarea (factor reward) entre episodios.
    base_env = GDEnvBase()
    env = ContinuousToDiscreteAction(TaskWrapper(base_env, task_id=0))

    while total_steps < args.timesteps:
        task_id = random.randrange(n_tasks)
        env.env.set_task(task_id)
        obs, _ = env.reset()
        ctx = agent._build_context(buffers[task_id])
        z, _ = agent.sample_z(ctx) if ctx is not None else (
            th.zeros(1, LATENT_DIM, device=DEVICE), 0.0)
        ep_r, ep_l, done = 0.0, 0, False

        while not done:
            a = agent.act(obs, z)
            obs2, r, term, trunc, _ = env.step(np.array([float(a[0])]))
            done = term or trunc
            buffers[task_id].push(obs, float(a[0]), r, obs2, done)
            obs = obs2; ep_r += r; ep_l += 1; total_steps += 1
            if total_steps % 4 == 0:
                agent.update(buffers)
            if total_steps >= args.timesteps:
                break

        ep += 1
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, total_steps, task_id, ep_r, ep_l])
        print(f"  ep {ep:4d}  task={task_id}  T={total_steps:6d}  "
              f"len={ep_l:4d}  r={ep_r:7.2f}")

        if ep % 50 == 0:
            agent.save(f"modelos_guardados/pearl_iter{ep}.pt")

    out = "modelos_guardados/pearl_FINAL.pt"
    agent.save(out)
    dt = time.perf_counter() - t0
    print(f"[PEARL] guardado -> {out}  ({dt:.0f}s,  {ep} episodios)")
    print(f"[PEARL] curvas   -> {csv_path}")


# ---------------------------------------------------------------------------
# 6. ADAPT (few-shot sobre cubo 2)
# ---------------------------------------------------------------------------
def adapt(args):
    print(f"[PEARL] adapt  base={args.base}  steps={args.adapt_steps}")
    print("  Coloca el juego en cubo 2 antes de continuar.")
    input("  Pulsa ENTER cuando este listo... ")

    agent = PEARL(n_tasks=1)
    agent.load(args.base)
    buf = TaskBuffer()
    env = ContinuousToDiscreteAction(TaskWrapper(GDEnvBase(), task_id=0))

    csv_path = f"metrics/pearl_adapt_{args.out_suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "timestep", "reward", "length"])

    obs, _ = env.reset()
    ep, ep_r, ep_l, total = 0, 0.0, 0, 0
    while total < args.adapt_steps:
        ctx = agent._build_context(buf)
        z, _ = agent.sample_z(ctx) if ctx is not None else (
            th.zeros(1, LATENT_DIM, device=DEVICE), 0.0)
        a = agent.act(obs, z)
        obs2, r, term, trunc, _ = env.step(np.array([float(a[0])]))
        done = term or trunc
        buf.push(obs, float(a[0]), r, obs2, done)
        obs = obs2; ep_r += r; ep_l += 1; total += 1
        if total % 4 == 0 and len(buf) > CONTEXT_LEN + 64:
            agent.update([buf])
        if done:
            ep += 1
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([ep, total, ep_r, ep_l])
            print(f"  ep {ep:3d}  T={total:5d}  len={ep_l:3d}  r={ep_r:6.2f}")
            obs, _ = env.reset(); ep_r, ep_l = 0.0, 0

    out = (f"modelos_guardados/pearl_{args.out_suffix}_adapted_"
           f"{args.adapt_steps}.pt")
    agent.save(out)
    print(f"[PEARL] adapt guardado -> {out}")


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["meta_train", "adapt"], required=True)
    ap.add_argument("--timesteps", type=int, default=60000)
    ap.add_argument("--base", default="modelos_guardados/pearl_FINAL.pt")
    ap.add_argument("--adapt_steps", type=int, default=8000)
    ap.add_argument("--out_suffix", type=str, default="cubo2",
                    help="sufijo para CSV y .pt (p.ej. cubo2, nivel2)")
    args = ap.parse_args()
    if args.mode == "meta_train":
        meta_train(args)
    else:
        adapt(args)
