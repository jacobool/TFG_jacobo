"""
Genera la figura de FRAME STACKING para la memoria: los 4 fotogramas
consecutivos (ya procesados a 84x84) que componen una observacion, mostrados
lado a lado. Es lo que ve la red gracias a VecFrameStack(n_stack=4).

Cada fotograma se procesa con el MISMO pipeline que gd_rl_env_4.py
(recorte ROI -> umbral 205 -> dilatacion 2x2 -> rectangulo jugador -> 84x84),
reutilizando generar_pipeline_completo.py.

Uso:
    # En vivo: captura 4 fotogramas consecutivos del juego en marcha.
    #   (lo ideal es lanzarlo con un episodio en curso para que haya movimiento)
    python generar_frame_stacking.py --live

    # A partir de 4 capturas ya guardadas, en orden:
    python generar_frame_stacking.py --imagenes f1.png f2.png f3.png f4.png

Salida por defecto: metrics/frame_stacking.png
    >>> SUBIR: metrics/frame_stacking.png -> imaxes/frame_stacking.png
"""

import argparse
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from generar_deteccion_jugador import capturar_en_vivo
from generar_pipeline_completo import construir_pipeline

N_STACK = 4
STEP_DURATION = 1 / 15  # mismo ritmo que el entorno


def procesar(frame_bgr, tol=12):
    """Devuelve la observacion final 84x84 de un fotograma."""
    return construir_pipeline(frame_bgr, tol=tol)["final"]


def capturar_secuencia(n, intervalo, tol, no_forzar):
    """Captura n fotogramas consecutivos y los procesa a 84x84."""
    obs = []
    for i in range(n):
        if no_forzar:
            frame = capturar_en_vivo(forzar_w=0, forzar_h=0)
        else:
            frame = capturar_en_vivo()
        obs.append(procesar(frame, tol=tol))
        print(f"[INFO] Fotograma {i + 1}/{n} capturado.")
        if i < n - 1:
            time.sleep(intervalo)
    return obs


def generar_figura(obs_list, salida):
    n = len(obs_list)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axs = [axs]
    for ax, obs in zip(axs, obs_list):
        ax.imshow(obs, cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura guardada en: {salida}")
    print(f">>> SUBIR: {salida} -> imaxes/{os.path.basename(salida)}")


def main():
    ap = argparse.ArgumentParser(description="Genera la figura de frame stacking (4 frames).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--live", action="store_true",
                     help="Capturar 4 fotogramas consecutivos del juego.")
    src.add_argument("--imagenes", nargs="+", metavar="IMG",
                     help="Rutas a los fotogramas ya guardados, en orden.")
    ap.add_argument("--salida", default="metrics/frame_stacking.png", help="Ruta de salida.")
    ap.add_argument("--tol", type=int, default=12,
                    help="Tolerancia HSV +-tol para el bbox del jugador (0 = rango exacto).")
    ap.add_argument("--intervalo", type=float, default=STEP_DURATION,
                    help="Segundos entre capturas en vivo (por defecto 1/15).")
    ap.add_argument("--no-forzar", action="store_true",
                    help="No forzar 800x600 en captura en vivo.")
    args = ap.parse_args()

    if args.live:
        obs_list = capturar_secuencia(N_STACK, args.intervalo, args.tol, args.no_forzar)
    else:
        obs_list = []
        for ruta in args.imagenes:
            frame = cv2.imread(ruta)
            if frame is None:
                raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
            obs_list.append(procesar(frame, tol=args.tol))

    generar_figura(obs_list, args.salida)


if __name__ == "__main__":
    main()
