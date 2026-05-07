# Archivo

Esta carpeta contiene **scripts y datos históricos** del proyecto que ya
no forman parte del flujo de trabajo principal pero **se conservan
intactos** por trazabilidad y para que la memoria pueda referenciarlos.

> ⚠️ **Nada de lo que está aquí se ha borrado**. Todo el contenido del
> proyecto sigue presente; solo se ha reorganizado para que la raíz del
> repositorio quede legible.

## Contenido por subcarpeta

| Subcarpeta | Contenido | Por qué se archivó |
|---|---|---|
| `capturas/` | `Captura de pantalla 2026-03-04 222530.png`, `attempt_template.png` | Capturas puntuales de depuración del módulo de visión, sin uso en el flujo actual. |
| `diagnostico_visual/` | `diag_ep01.png`–`diag_ep10.png`, `diag_donde_muere.png`, `diag_todos_frames.csv`, `diagnostico_deteccion.csv`, `diagnostico_deteccion.png` | Salidas del *script* de diagnóstico de detección de muerte y del muestreo de fotogramas durante la calibración inicial. |
| `evaluaciones_antiguas/` | `eval_resultados.csv`, `eval_resultados_2.csv`, `eval_comparativa.png`, `eval_comparativa_2.png`, `comparacion_modelos.png` | Evaluaciones comparativas anteriores al estudio comparativo formal del cap. 5 de la memoria. Se conservan por si hay que rescatar gráficas. |
| `resultados_progressive_v0/` | `resultados_progressive.csv`, `resultados_progressive_2.csv` | Resultados intermedios del primer experimento de Progressive Networks (antes del fix del *deadlock* de inicialización de las escalas laterales). |
| `scripts_envs_antiguos/` | `gd_rl_env_1.py`–`gd_rl_env_3.py`, `gd_rl_env_5.py` | Iteraciones tempranas del entorno `GDEnv`. La versión activa es `gd_rl_env_4.py` en la raíz. **Nadie en activo importa estos.** |
| `scripts_ppo_rppo/` | `gd_rl_env_4_ppo.py`, `gd_rl_env_4_ppo_v2.py`, `gd_rl_env_4_ppo_v3.py`, `gd_rl_env_4_recurrentppo.py` | Versiones PPO y Recurrent PPO sobre el cubo 1. Tras el estudio comparativo se eligió QR-DQN como motor del sistema; estos *scripts* documentan los entrenamientos cuyas curvas aparecen en `metrics/ppo_*.csv` y `metrics/recurrentppo_*.csv`. |
| `scripts_nave_antiguos/` | `gd_rl_nave_1.py`, `gd_rl_nave_1-2(parte_800k).py`, `gd_rl_nave_2_seguirentreno.py` | Iteraciones DQN de la nave previas a la versión QR-DQN. La activa es `gd_rl_nave_3_qrdqn.py` en la raíz. **`gd_rl_nave_2.py` se ha quedado en la raíz** porque `jugar_nave_eval.py` (activo) lo importa. |
| `scripts_jugar_antiguos/` | `jugar_gd_2.py`, `jugar_gd_3.py`, `jugar_nave_1.py`, `jugar_nave_2.py` | Versiones anteriores de los *scripts* de inferencia. La versión actual es `jugar_gd_4.py` (cubo, con autodetección QR-DQN/DQN), `jugar_todo.py` (orquestador multimodal) y `jugar_nave_3.py`. |
| `scripts_deteccion_muerte/` | `detectar_muerte.py`, `detectar_muertev2.py`, `detectar_mismo.py` | Pruebas tempranas del detector de muerte. La versión productiva es `gd_death_detector.py` en la raíz, integrada en `GDEnv`. |
| `scripts_experimentos_iniciales/` | `geo_dqn_4frames.py`, `geometry_dash_v2.py`, `geometry_prueba.py`, `main.py`, `pr.py` | *Scripts* de exploración inicial del problema, sin relación directa con el sistema final. Se mantienen por documentación histórica. |

## Cómo recuperar un archivo

Si en algún momento necesitas resucitar un *script* archivado:

```bash
mv archivo/<subcarpeta>/<archivo>.py .
```

Y atención a las dependencias: algunos *scripts* archivados importan
otros del mismo grupo. Si mueves uno, comprueba si tiene `from gd_rl_*`
o `from jugar_*` dentro del fichero antes de ejecutarlo.

## Lo que NO se ha movido

Permanecen en la raíz **todos** los *scripts* activos del estudio
comparativo, los modelos `.zip` finales referenciados por *scripts* de
inferencia, y los CSV/PNG que aún se actualizan (los de `metrics/`
también se quedan donde están). Consulta `GUIA.md` en la raíz para el
mapa completo.
