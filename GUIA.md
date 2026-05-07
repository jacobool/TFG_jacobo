# Guía de navegación del proyecto

Este documento es el **mapa del repositorio**: explica qué hay en cada
carpeta, qué *scripts* activos existen, en qué orden ejecutarlos y qué
artefactos genera cada uno. Para el contexto del proyecto consulta
`CLAUDE.md` (visión general) y `memoria/` (documento académico).

> 🛑 **Nada se ha borrado.** Todo lo histórico está en `archivo/`.

## Estructura del repositorio

```
TFG/
├── CLAUDE.md                    Contexto del proyecto (lee Claude Code)
├── GUIA.md                      Este fichero
├── explicacion_ewc.md           Notas técnicas detalladas sobre EWC
├── requirements.txt
│
├── *.py                         Scripts activos (35 ficheros, ver más abajo)
├── gd_metrics.csv               Métricas DQN cubo 1 (entrenamiento base, ~500k pasos)
├── gd_ppo_v2_FINAL_4.zip        Modelo PPO v2 cubo 1 (con vecnormalize.pkl)
├── gd_recurrentppo_FINAL_4.zip  Modelo RecurrentPPO cubo 1
│
├── archivo/                     Scripts y datos históricos (ver archivo/README.md)
│   ├── capturas/
│   ├── diagnostico_visual/
│   ├── evaluaciones_antiguas/
│   ├── resultados_progressive_v0/
│   ├── scripts_deteccion_muerte/
│   ├── scripts_envs_antiguos/
│   ├── scripts_experimentos_iniciales/
│   ├── scripts_jugar_antiguos/
│   ├── scripts_nave_antiguos/
│   └── scripts_ppo_rppo/
│
├── memoria/                     Documento académico (LaTeX) y anteproyecto
│   ├── Anteproxecto-TFG-Jacobo Olmedo Sánchez.pdf
│   ├── memoria_tfg_v3.zip       ← último ZIP listo para subir a Prism
│   └── extracted/               Fuentes LaTeX descomprimidas (de trabajo)
│
├── metrics/                     Métricas y plots de entrenamientos activos
├── modelos_guardados/           Checkpoints y modelos finales (entrenamientos recientes)
├── models/                      Modelos antiguos (DQN cubo 1 base + nave DQN)
├── plots/                       Plots auxiliares (sólo de la nave inicial)
│
├── venv_geometry/               Virtualenv (no tocar)
├── __pycache__/                 Caché de Python (no tocar)
└── .claude/                     Configuración Claude Code (no tocar)
```

## Catálogo de *scripts* activos por categoría

### Entornos y utilidades base (no ejecutar directamente, sólo se importan)

| Script | Función | Importado por |
|---|---|---|
| `gd_rl_env_4.py` | Define `GDEnv` (entorno Gymnasium con captura de pantalla, detección morfológica HSV, recompensa). Es el **núcleo** que usa todo lo demás. | Todos los `gd_rl_env_4_qrdqn_*`, `generar_replay_cubo1.py`, `evaluar_cubo1_postadaptacion.py`, `vuelo_ver_muerte.py` |
| `gd_rl_cubo2_progressive.py` | Define `ProgressiveCNN` y `GDEnvCubo`. Si lo ejecutas directamente, entrena PNN sobre DQN. | `gd_rl_cubo2_progressive_qrdqn.py` |
| `gd_rl_nave_2.py` | Entorno antiguo de la nave (sin tocar — `jugar_nave_eval.py` lo importa). | `jugar_nave_eval.py` |
| `gd_death_detector.py` | Detector de muerte robusto basado en doble señal (motion stop + texto "PT X"). | (uso interno de `GDEnv`) |
| `gd_rl_completo.py` | Entorno multimodal cubo+nave para jugar el nivel entero. | (uso opcional de `jugar_todo.py`) |

### Entrenamiento — agentes base sobre cubo 1 (~500k pasos cada uno)

| Script | Algoritmo | Salida |
|---|---|---|
| `gd_rl_env_4_qrdqn.py` | **QR-DQN** (motor del sistema final) | `modelos_guardados/gd_qrdqn_*_steps.zip`, `metrics/qrdqn_metrics.csv` |

> Las versiones DQN, PPO y RecurrentPPO están en `archivo/scripts_ppo_rppo/`.
> Los CSV de sus entrenamientos (que sí están vivos) siguen en `metrics/`.

### Entrenamiento — agente nave (~500k pasos)

| Script | Algoritmo | Salida |
|---|---|---|
| `gd_rl_nave_3_qrdqn.py` | **QR-DQN** (motor unificado del sistema) | `modelos_guardados/nave_qrdqn_FINAL.zip`, `metrics/nave_metrics_QRDQN.csv` |

### Entrenamiento — adaptación al cubo 2 (siete estrategias del estudio comparativo)

| # | Script | Estrategia | Algoritmo | Estado |
|---|---|---|---|---|
| 1 | `gd_rl_env_4_qrdqn_scratch_cubo2.py` | Tabula rasa | QR-DQN | ✅ ejecutado (80k) |
| 2 | `gd_rl_env_4_qrdqn_finetune_cubo2.py` | Fine-tune ingenuo | QR-DQN | ✅ ejecutado (80k) |
| 3 | `gd_rl_env_4_qrdqn_distill_cubo2.py` | Policy distillation | QR-DQN | ✅ ejecutado (80k) |
| 4 | `gd_rl_env_4_qrdqn_replay_cubo2.py` | **Replay buffer mixing** | QR-DQN | 🟡 listo (600k) |
| 5 | `gd_rl_env_4_qrdqn_ewc_cubo2.py` | **EWC sobre QR-DQN** | QR-DQN | 🟡 listo (600k) |
| 6 | `gd_rl_cubo2_progressive_qrdqn.py` | **PNN sobre QR-DQN** | QR-DQN | 🟡 listo (600k) |
| 7 | `gd_rl_cubo2_ewc.py` (+ `_seguirentreno.py`) | EWC sobre DQN | DQN | ✅ ejecutado (600k, contraste) |
| 8 | `gd_rl_cubo2_progressive.py` | PNN sobre DQN | DQN | 🟡 reentreno pendiente con fix del *deadlock* |
| 9 | `generar_replay_cubo1.py` | Helper: genera `models/replay_cubo1.pkl` para la rama 4 | (no es estrategia) | ✅ ejecutado |

### Inferencia y evaluación

| Script | Para qué sirve | Salida |
|---|---|---|
| `jugar_gd_4.py` | Reproduce un modelo cubo en GD. Autodetecta QR-DQN vs DQN por nombre del *zip*. **Cambia `MODEL_PATH` para elegir el modelo.** | (no genera fichero; muestra al agente jugando) |
| `jugar_todo.py` | Orquestador multi-agente con conmutación cubo↔nave por histéresis. | (no genera fichero) |
| `jugar_nave_3.py` | Reproduce el agente QR-DQN de la nave. | — |
| `jugar_nave_eval.py` | Evaluación cuantitativa del agente nave (varias partidas, estadísticas). | `eval_*.csv` |
| `jugar_gd_4_cubo2.py` | Variante de `jugar_gd_4.py` con posición inicial en cubo 2. | — |
| `jugar_gd_4_ppo.py` | Reproduce el modelo PPO v2 cubo 1 (`gd_ppo_v2_FINAL_4.zip`). | — |
| `jugar_gd_cubo2_progressive.py` | Reproduce un modelo PNN. **Importa `ProgressiveCNN`** de `gd_rl_cubo2_progressive.py`. | — |
| `evaluar_cubo1_postadaptacion.py` | **Métrica clave del estudio comparativo**: para cada modelo de adaptación juega 20 partidas en cubo 1 con `deterministic=True`, reporta tasa de supervivencia. | `metrics/eval_cubo1_postadapt.csv` y `_resumen.csv` |
| `evaluar_modelos.py` | Evaluación comparativa entre modelos (legacy, anterior al estudio formal). | — |
| `comparar_cubo2.py` | Genera la gráfica comparativa de las estrategias de cubo 2 a partir de los CSV de `metrics/`. | `metrics/comparativa_cubo2.png` |

### Calibración de visión y diagnóstico

| Script | Función |
|---|---|
| `calibrate_gd.py` | Calibración HSV del cubo (modo cubo 1). |
| `calibrate_nave.py` | Calibración HSV de la nave. |
| `calibrar_cubo_a_nave.py` | Ajuste del *offset* de detección al cambiar de modo. |
| `calibrar_fondo_red.py` | Calibración del umbral de blanco para la rama de obstáculos. |
| `detectar_color.py` | Inspección rápida del color de un píxel (depuración). |
| `diagnostico_deteccion.py` | Visualización paso a paso de la detección morfológica del jugador. |
| `ver_obs_gdenv2.py` | Visualizador en directo de la observación que recibe la red (84×84 binaria). |
| `vuelo_ver_muerte.py` | Diagnóstico del detector de muerte (visualiza señales). |
| `limpiar_recurrentppo.py` | Filtra *outliers* por congelaciones de ventana en los CSV de RecurrentPPO. |

## Plan de ejecución pendiente (prioridad alta → baja)

1. 🟡 **`gd_rl_env_4_qrdqn_replay_cubo2.py`** — Replay buffer mixing 600k.
2. 🟡 **`gd_rl_env_4_qrdqn_ewc_cubo2.py`** — EWC sobre QR-DQN 600k.
3. 🟡 **`gd_rl_cubo2_progressive.py`** — Reentreno PNN DQN 600k con el *fix* de inicialización.
4. 🟡 **`gd_rl_cubo2_progressive_qrdqn.py`** — PNN sobre QR-DQN 600k.
5. 🔴 **`evaluar_cubo1_postadaptacion.py`** — ejecutar después de cada uno de los anteriores. **Es la métrica decisiva** del estudio comparativo.

A 10 *fps* reales, cada *run* de 600k son ~17 horas de juego. La
evaluación cubo 1 son 20 partidas × N modelos ≈ 30 minutos totales.

## Carpetas que se actualizan durante el trabajo

- **`metrics/`**: cada *script* de entrenamiento escribe aquí su CSV y su PNG. Es la carpeta de salida principal.
- **`modelos_guardados/`**: *checkpoints* periódicos y `*_FINAL.zip` de cada *run*.
- **`memoria/extracted/`**: fuentes LaTeX descomprimidas que voy editando. Para subir a Prism, recomprimir como `memoria_tfg_v3.zip`.

## Convenciones de nombres

| Patrón | Significado |
|---|---|
| `gd_rl_env_4_qrdqn_<estrategia>_cubo2.py` | Adaptación al cubo 2 sobre QR-DQN |
| `gd_rl_cubo2_<estrategia>.py` | Adaptación al cubo 2 sobre DQN clásico |
| `gd_rl_cubo2_<estrategia>_qrdqn.py` | Mismo, pero migrado a QR-DQN |
| `gd_rl_nave_<n>.py` | Iteración n del entrenamiento de nave (la 3 es QR-DQN, definitiva) |
| `jugar_<modelo>.py` | *Script* de inferencia/reproducción de un modelo concreto |

## Cómo encontrar rápido un fichero

| Busco… | Está en… |
|---|---|
| El entorno principal `GDEnv` | `gd_rl_env_4.py` (raíz) |
| El profesor del cubo 1 | `modelos_guardados/gd_qrdqn_440000_steps.zip` |
| La gráfica del entreno QR-DQN cubo 1 | `metrics/qrdqn_plot.png` |
| Las métricas de un *run* concreto | `metrics/<nombre_del_run>_metrics.csv` |
| El modelo final de una rama | `modelos_guardados/<nombre>_FINAL.zip` |
| Un *script* "viejo" que recuerdo haber visto | `archivo/` (subcarpeta temática) |
| El último ZIP de la memoria | `memoria/memoria_tfg_v3.zip` |
| Los .tex de la memoria que estoy editando | `memoria/extracted/modelo-tfg-fic-v1.6_2223xun/contido/*.tex` |
