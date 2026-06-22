"""
Genera las dos observaciones para la figura de evolucion del entorno:

    v2 -> deteccion de bordes (Canny): el avatar se pierde entre los contornos
    v3 -> binarizacion por umbral + rectangulo blanco sobre el jugador

Reproduce EXACTAMENTE los _get_obs de los scripts archivados:
    archivo/scripts_envs_antiguos/gd_rl_env_2.py  (v2)
    archivo/scripts_envs_antiguos/gd_rl_env_3.py  (v3)

Lo ideal es partir de UNA misma captura para que la comparacion sea justa:
    python generar_obs_v2_v3.py --imagen captura.png

O capturar en vivo (juego abierto, mejor con una partida en marcha):
    python generar_obs_v2_v3.py --live

Salidas por defecto (las dos que pide el .tex):
    metrics/obs_bordes_v2.png      >>> SUBIR -> imaxes/obs_bordes_v2.png
    metrics/obs_binarizada_v3.png  >>> SUBIR -> imaxes/obs_binarizada_v3.png
"""

import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from generar_deteccion_jugador import capturar_en_vivo

# --- Constantes de la epoca de v2/v3 (gd_rl_env_2.py / gd_rl_env_3.py) ---
PLAYER_X_REL = 0.37
PLAYER_BAND_W = 0.06
LOWER_GREEN = np.array([46, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([46, 255, 255], dtype=np.uint8)


def detectar_bbox(frame_bgr, tol=12):
    """bbox del jugador via deteccion HSV (parametros de v2/v3)."""
    h, w = frame_bgr.shape[:2]
    player_x = int(w * PLAYER_X_REL)
    band_half = int(w * PLAYER_BAND_W / 2)
    x1 = max(0, player_x - band_half)
    x2 = min(w, player_x + band_half)
    y1, y2 = int(h * 0.08), int(h * 0.92)
    band_bgr = frame_bgr[y1:y2, x1:x2]

    lower = np.clip(LOWER_GREEN.astype(int) - tol, 0, 255).astype(np.uint8)
    upper = np.clip(UPPER_GREEN.astype(int) + tol, 0, 255).astype(np.uint8)

    if band_bgr.size == 0:
        return None
    band_hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(band_hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = max(contours, key=cv2.contourArea) if contours else None
    if best is not None and cv2.contourArea(best) > 50:
        bx, by, bw, bh = cv2.boundingRect(best)
        return (x1 + bx, y1 + by, bw, bh)
    return None


def obs_v2(frame_bgr):
    """Observacion v2 (deteccion de bordes), EXACTA segun ver_obs_gdenv2.py:
    recorte 20-80% gris, blur(3,3), Canny(170,270), dilatacion 3x3 y resize.
    Es la version que se visualizaba realmente: la dilatacion engrosa los
    bordes hasta que el avatar se pierde entre el fondo."""
    h, w = frame_bgr.shape[:2]
    x1, x2 = int(w * 0.20), int(w * 0.80)
    cropped = cv2.cvtColor(frame_bgr[:, x1:x2], cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(cropped, (3, 3))
    edges = cv2.Canny(blurred, threshold1=170, threshold2=270)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.resize(dilated, (84, 84))


def obs_v3(frame_bgr, tol=12):
    """Observacion v3: recorte 20-90% gris, umbral 205, dilatacion 2x2 y
    rectangulo blanco solido sobre el jugador."""
    h, w = frame_bgr.shape[:2]
    x1_ai, x2_ai = int(w * 0.20), int(w * 0.90)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cropped = gray[:, x1_ai:x2_ai]

    _, thresh = cv2.threshold(cropped, 205, 255, cv2.THRESH_BINARY)
    vision = cv2.dilate(thresh, np.ones((2, 2), np.uint8), iterations=1)

    bbox = detectar_bbox(frame_bgr, tol=tol)
    if bbox is not None:
        x_abs, y_abs, bw, bh = bbox
        p_x1 = x_abs - x1_ai
        if p_x1 >= 0:
            cv2.rectangle(vision, (p_x1, y_abs), (p_x1 + bw, y_abs + bh), 255, -1)
    return cv2.resize(vision, (84, 84)), bbox


def guardar(obs, salida):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(obs, cmap="gray")
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
    plt.savefig(salida, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[OK] Figura guardada en: {salida}")
    print(f">>> SUBIR: {salida} -> imaxes/{os.path.basename(salida)}")


def main():
    ap = argparse.ArgumentParser(description="Genera observaciones v2 (bordes) y v3 (binarizada).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--imagen", help="Captura/fotograma ya guardado (recomendado, misma para ambas).")
    src.add_argument("--live", action="store_true", help="Capturar de la ventana del juego.")
    ap.add_argument("--salida-v2", default="metrics/obs_bordes_v2.png")
    ap.add_argument("--salida-v3", default="metrics/obs_binarizada_v3.png")
    ap.add_argument("--tol", type=int, default=12,
                    help="Tolerancia HSV +-tol para el bbox de v3 (0 = rango exacto).")
    ap.add_argument("--no-forzar", action="store_true",
                    help="No forzar 800x600 en captura en vivo.")
    ap.add_argument("--guardar-captura", metavar="RUTA",
                    help="Guarda el fotograma capturado en crudo.")
    args = ap.parse_args()

    if args.live:
        frame = capturar_en_vivo(forzar_w=0, forzar_h=0) if args.no_forzar else capturar_en_vivo()
        if args.guardar_captura:
            cv2.imwrite(args.guardar_captura, frame)
            print(f"[OK] Captura guardada en: {args.guardar_captura}")
    else:
        frame = cv2.imread(args.imagen)
        if frame is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {args.imagen}")

    guardar(obs_v2(frame), args.salida_v2)
    obs3, bbox = obs_v3(frame, tol=args.tol)
    if bbox is None:
        print("[AVISO] v3: no se detecto al jugador; sin rectangulo. Prueba --tol 25.")
    guardar(obs3, args.salida_v3)


if __name__ == "__main__":
    main()
