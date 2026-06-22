# Geometry Dash RL — Agente de Reinforcement Learning

Agente de aprendizaje por refuerzo que juega a **Geometry Dash** de forma autónoma. Observa el juego en tiempo real mediante visión por computador, toma decisiones con redes neuronales profundas y controla el teclado para esquivar obstáculos.

El proyecto cubre dos modos de juego — **Cubo** y **Nave (Ship)** — e incluye una batería de experimentos de **transfer learning** para que el agente aprenda nuevos niveles sin olvidar los anteriores.

> Trabajo de Fin de Grado — Universidade da Coruña (UDC)

---

## Cómo funciona

```
┌─────────────┐    captura     ┌──────────────┐    observación    ┌────────────┐
│  Geometry    │ ─────────────►│  Visión por   │ ────────────────►│  Agente    │
│  Dash        │   (mss+cv2)   │  computador   │   (84×84 bin)    │  DQN/QRDQN │
│  (ventana)   │◄──────────────│  (HSV + bordes)│◄─────────────── │  (PyTorch) │
└─────────────┘   pydirectinput└──────────────┘    acción: pulsar │            │
                  (teclado)                        o soltar tecla  └────────────┘
```

1. **Captura de pantalla** a ~15 FPS con `mss` (sin pérdida de rendimiento).
2. **Detección del jugador** en espacio HSV — localiza el blob de color del personaje y extrae su posición (x, y) y tamaño.
3. **Observación binaria 84×84**: se filtran los píxeles blancos (obstáculos), se superpone el hitbox del jugador, y se apilan los 4 últimos frames para dar contexto temporal.
4. **Decisión**: el agente elige entre dos acciones — mantener pulsado espacio (subir/saltar) o soltarlo (caer/no saltar).
5. **Recompensa**: +reward por cada frame vivo, bonus por batir récord de distancia, penalización fuerte al morir, pequeñas penalizaciones por cambios bruscos de acción y por volar demasiado cerca de los bordes.
6. **Detección de muerte**: combinación de parada del movimiento del fondo y aparición del texto de intento ("PT X"). Independiente del color del personaje.

## Requisitos

- **Windows 10/11** (usa DirectInput para el teclado y APIs de ventana de Windows)
- **Python 3.10+**
- **Geometry Dash** instalado (Steam) y visible en pantalla
- GPU con CUDA recomendada (entrena con PyTorch)

## Instalación

```bash
git clone https://github.com/tu-usuario/geometry-dash-rl.git
cd geometry-dash-rl

python -m venv venv_geometry
source venv_geometry/Scripts/activate   # Git Bash / WSL
# o: venv_geometry\Scripts\activate     # PowerShell / cmd

pip install -r requirements.txt
```

## Uso rápido

Todos los scripts se ejecutan desde la **raíz del proyecto**. Asegúrate de que Geometry Dash esté abierto y visible antes de lanzar cualquier script.

### Entrenar

```bash
# Modo Nave (entrenamiento principal)
python src/core/gd_rl_nave_2.py

# Modo Cubo (DQN base)
python src/core/gd_rl_env_4.py

# Modo Cubo con QR-DQN
python src/core/gd_rl_env_4_qrdqn.py

# Juego completo (cubo + nave en un solo episodio)
python src/training/gd_rl_completo.py
```

Los checkpoints se guardan automáticamente en `modelos_guardados/` cada N pasos. Los modelos finales se copian a `models/`.

### Jugar con un modelo entrenado

```bash
# Nave (QR-DQN)
python src/inference/jugar_nave_3.py

# Cubo
python src/inference/jugar_gd_4.py

# Cualquier modelo (detecta el tipo automáticamente)
python src/inference/jugar_universal.py --model models/gd_dqn_FINAL_4
```

### Evaluar checkpoints

```bash
# Comparar checkpoints del modo nave
python src/inference/jugar_nave_eval.py

# Evaluador general con estadísticas
python src/evaluation/evaluar_modelos.py --model models/gd_dqn_FINAL_4 --episodes 50
```

### Calibrar la visión

Si cambias el skin del personaje en Geometry Dash, los umbrales HSV deben recalibrarse:

```bash
python src/tools/calibrate_nave.py        # Calibrar color del jugador (nave)
python src/tools/calibrate_gd.py          # Calibrar color del jugador (cubo)
python src/tools/calibrar_cubo_a_nave.py  # Calibrar offset cubo → nave
```

### Depurar

```bash
python src/tools/ver_obs_gdenv2.py     # Ver lo que ve el agente en tiempo real
python src/tools/vuelo_ver_muerte.py   # Visualizar señales de detección de muerte
```

## Experimentos de Transfer Learning

El proyecto investiga cómo transferir conocimiento de un nivel a otro sin olvidar lo aprendido. Todos los scripts están en `src/training/`:

| Técnica | Script | Descripción |
|---------|--------|-------------|
| **EWC** (Elastic Weight Consolidation) | `gd_rl_cubo2_ewc.py` | Penaliza cambios en pesos importantes del nivel anterior |
| **Progressive Networks** | `gd_rl_cubo2_progressive.py` | Columna congelada del nivel 1 + columna nueva con conexiones laterales |
| **Fine-tuning** | `gd_rl_env_4_qrdqn_finetune_cubo2.py` | Continúa entrenando directamente en el nuevo nivel |
| **Knowledge Distillation** | `gd_rl_env_4_qrdqn_distill_cubo2.py` | Destila conocimiento del modelo profesor al alumno |
| **Experience Replay** | `gd_rl_env_4_qrdqn_replay_cubo2.py` | Mezcla experiencias del nivel anterior con el nuevo |
| **Scratch** (baseline) | `gd_rl_env_4_qrdqn_scratch_cubo2.py` | Entrena desde cero como referencia |
| **MAML** | `gd_rl_maml_cubo.py` | Meta-learning para adaptación rápida |
| **PEARL** | `gd_rl_pearl_cubo.py` | Meta-RL con inferencia de contexto latente |
| **SAC** | `gd_rl_sac_cubo.py` | Soft Actor-Critic (acciones continuas) |

## Estructura del proyecto

```
├── src/                        Código fuente
│   ├── core/                   Entornos Gymnasium + detector de muerte
│   ├── training/               Scripts de entrenamiento y transfer learning
│   ├── inference/              Scripts para jugar con modelos entrenados
│   ├── evaluation/             Evaluación y comparación de modelos
│   ├── tools/                  Calibración HSV y herramientas de debug
│   ├── plotting/               Generación de gráficas y figuras
│   └── utils/                  Utilidades varias
│
├── archivo/                    Scripts antiguos archivados (por categoría)
├── modelos_guardados/          Checkpoints de entrenamiento (.zip)
├── models/                     Modelos finales/mejores de cada experimento
├── metrics/                    Métricas CSV y gráficas de entrenamiento
├── plots/                      Gráficas de salida generadas
├── memoria/                    Documento del TFG
├── docs/                       Documentación adicional
│
├── setup_paths.py              Configuración de rutas (importado por todos los scripts)
├── requirements.txt            Dependencias Python
└── venv_geometry/              Entorno virtual (no incluido en git)
```

### Sobre `setup_paths.py`

Cada script dentro de `src/` importa `setup_paths` al inicio. Este módulo añade la raíz del proyecto y todas las subcarpetas de `src/` al `sys.path` de Python, y establece el directorio de trabajo en la raíz. Esto permite que:
- Los imports cruzados entre módulos funcionen sin importar la ubicación del script.
- Las rutas relativas a datos (`modelos_guardados/`, `models/`, `metrics/`) se resuelvan correctamente.

## Stack tecnológico

| Componente | Librería |
|------------|----------|
| RL | `stable_baselines3` 2.7.1 (DQN, QR-DQN, PPO, SAC) |
| Deep Learning | `torch` 2.10.0 + CUDA |
| Entorno Gym | `gymnasium` 1.2.3 |
| Visión | `opencv-python` 4.13.0 |
| Captura de pantalla | `mss` 10.1.0 |
| Control de teclado | `pydirectinput` 1.0.4 (DirectInput) |
| Gestión de ventana | `pygetwindow` + `ctypes` Win32 API |

## Notas importantes

- El juego **debe estar visible en pantalla** y en primer plano. Los scripts localizan la ventana por título con `pygetwindow`.
- Los umbrales HSV de detección están ajustados a skins específicos del juego. Si cambias de skin, usa los scripts de calibración en `src/tools/`.
- `pydirectinput` envía inputs DirectInput, compatibles con juegos en fullscreen. Los scripts fuerzan el foco de la ventana automáticamente con `ctypes`.
- Los comentarios y nombres de variables en el código están principalmente en **español**.
- El entrenamiento es en tiempo real: el agente juega partidas reales, no en simulación. Cada episodio dura lo que el agente sobrevive en el juego.
