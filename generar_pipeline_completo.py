"""
Genera la figura del PIPELINE COMPLETO de percepcion para la memoria:

    captura cruda -> recorte ROI -> umbralizacion -> dilatacion -> 84x84

Reproduce EXACTAMENTE gd_rl_env_4.py (_capture_frame + _get_obs):
    - ROI gris recortando al 20%-90% del ancho
    - umbral binario a 205
    - dilatacion con kernel 2x2 (1 iteracion)
    - rectangulo blanco solido sobre el jugador (bbox de la deteccion HSV)
    - redimension final a 84x84 con INTER_AREA

Uso:
    # A partir de una captura ya guardada:
    python generar_pipeline_completo.py --imagen captura.png

    # Capturando en vivo desde la ventana del juego (debe estar abierto):
    python generar_pipeline_completo.py --live

Salida por defecto: metrics/pipeline_completo.png
    >>> SUBIR: metrics/pipeline_completo.png -> imaxes/pipeline_completo.png
"""

import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Reutilizamos captura (con fix DPI/800x600) y deteccion HSV del otro script
from generar_deteccion_jugador import capturar_en_vivo, detectar

# Constantes del recorte de observacion (gd_rl_env_4.py)
ROI_X1_REL = 0.20
ROI_X2_REL = 0.90
THRESH_VAL = 205
DILATE_KERNEL = np.ones((2, 2), np.uint8)


def construir_pipeline(frame_bgr, tol=12):
    """Devuelve los fotogramas intermedios del pipeline replicando el entorno."""
    h, w = frame_bgr.shape[:2]

    # --- bbox del jugador via deteccion HSV (misma banda que el env) ---
    _, bbox_abs, _ = detectar(frame_bgr, tol=tol)

    # --- 1. recorte ROI en gris (img[:, x1_ai:x2_ai]) ---
    x1_ai, x2_ai = int(w * ROI_X1_REL), int(w * ROI_X2_REL)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    roi = gray[:, x1_ai:x2_ai]

    # --- 2. umbralizacion binaria a 205 ---
    _, thresh = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)

    # --- 3. dilatacion (kernel 2x2) ---
    dilated = cv2.dilate(thresh, DILATE_KERNEL, iterations=1)

    # --- rectangulo blanco solido sobre el jugador ---
    dilated_player = dilated.copy()
    if bbox_abs is not None:
        x_abs, y_abs, bw, bh = bbox_abs
        p_x1 = x_abs - x1_ai
        p_y1 = y_abs
        if p_x1 >= 0:
            cv2.rectangle(dilated_player, (p_x1, p_y1), (p_x1 + bw, p_y1 + bh), 255, -1)

    # --- 4. redimension final a 84x84 ---
    final = cv2.resize(dilated_player, (84, 84), interpolation=cv2.INTER_AREA)

    return {
        "captura": frame_bgr,
        "roi": roi,
        "thresh": thresh,
        "dilated": dilated_player,
        "final": final,
        "roi_x": (x1_ai, x2_ai),
        "bbox_abs": bbox_abs,
    }


def generar_figura(pasos, salida):
    frame_rgb = cv2.cvtColor(pasos["captura"], cv2.COLOR_BGR2RGB)
    x1_ai, x2_ai = pasos["roi_x"]

    # Panel 1: captura con la franja ROI marcada
    panel_cap = frame_rgb.copy()
    cv2.rectangle(panel_cap, (x1_ai, 0), (x2_ai, panel_cap.shape[0] - 1), (255, 200, 0), 2)

    fig, axs = plt.subplots(1, 5, figsize=(22, 6))

    axs[0].imshow(panel_cap)
    axs[1].imshow(pasos["roi"], cmap="gray")
    axs[2].imshow(pasos["thresh"], cmap="gray")
    axs[3].imshow(pasos["dilated"], cmap="gray")
    axs[4].imshow(pasos["final"], cmap="gray")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura guardada en: {salida}")
    print(f">>> SUBIR: {salida} -> imaxes/{os.path.basename(salida)}")


def main():
    ap = argparse.ArgumentParser(description="Genera la figura del pipeline de percepcion.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--imagen", help="Ruta a una captura/fotograma ya guardado.")
    src.add_argument("--live", action="store_true", help="Capturar de la ventana del juego.")
    ap.add_argument("--salida", default="metrics/pipeline_completo.png", help="Ruta de salida.")
    ap.add_argument("--tol", type=int, default=12,
                    help="Tolerancia HSV +-tol para el bbox del jugador (0 = rango exacto).")
    ap.add_argument("--no-forzar", action="store_true",
                    help="No forzar 800x600 en captura en vivo.")
    ap.add_argument("--guardar-captura", metavar="RUTA",
                    help="Guarda el fotograma capturado en crudo (para verificar encuadre).")
    args = ap.parse_args()

    if args.live:
        if args.no_forzar:
            frame = capturar_en_vivo(forzar_w=0, forzar_h=0)
        else:
            frame = capturar_en_vivo()
        if args.guardar_captura:
            cv2.imwrite(args.guardar_captura, frame)
            print(f"[OK] Captura guardada en: {args.guardar_captura}")
    else:
        frame = cv2.imread(args.imagen)
        if frame is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {args.imagen}")

    pasos = construir_pipeline(frame, tol=args.tol)
    if pasos["bbox_abs"] is None:
        print("[AVISO] No se detecto al jugador; el rectangulo no aparecera. Prueba --tol 25.")
    generar_figura(pasos, args.salida)


if __name__ == "__main__":
    main()
