import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

"""
Genera la figura de 3 paneles para la memoria:
    fotograma original  ->  mascara verde HSV  ->  contorno detectado

Reproduce EXACTAMENTE la deteccion de gd_rl_env_4.py:
    - banda vertical centrada en el jugador (PLAYER_X_REL / PLAYER_BAND_W)
    - mascara HSV con LOWER_GREEN / UPPER_GREEN
    - apertura + cierre morfologico (kernel 3x3)
    - contorno principal (mayor area) + bounding box

Uso:
    # 1) A partir de una captura ya guardada (lo mas comodo):
    python generar_deteccion_jugador.py --imagen captura.png

    # 2) Capturando en vivo desde la ventana del juego (debe estar abierto):
    python generar_deteccion_jugador.py --live

Notas:
    - El rango HSV del proyecto es muy estricto ([45,255,255]). Para que la
      mascara se vea bien sobre una captura real (con antialiasing) se aplica
      una tolerancia por defecto (--tol). Pon --tol 0 para usar el rango exacto.
    - Salida por defecto: metrics/deteccion_jugador.png
      >>> SUBIR: metrics/deteccion_jugador.png -> imaxes/deteccion_jugador.png
"""

import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Constantes copiadas de gd_rl_env_4.py ---
GAME_TITLE = "Geometry Dash"
PLAYER_X_REL = 0.345
PLAYER_BAND_W = 0.065
LOWER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([45, 255, 255], dtype=np.uint8)


def detectar(frame_bgr, tol=0):
    """Devuelve (mascara_banda, bbox_abs, banda_coords) replicando el entorno.

    tol ensancha el rango HSV +-tol en cada canal (solo para visualizacion).
    """
    h, w = frame_bgr.shape[:2]
    player_x = int(w * PLAYER_X_REL)
    band_half = int(w * PLAYER_BAND_W / 2)
    x1 = max(0, player_x - band_half)
    x2 = min(w, player_x + band_half)
    y1, y2 = int(h * 0.08), int(h * 0.92)
    band_bgr = frame_bgr[y1:y2, x1:x2]

    lower = np.clip(LOWER_GREEN.astype(int) - tol, 0, 255).astype(np.uint8)
    upper = np.clip(UPPER_GREEN.astype(int) + tol, 0, 255).astype(np.uint8)

    mask_full = np.zeros((h, w), dtype=np.uint8)
    bbox_abs = None

    if band_bgr.size > 0:
        band_hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(band_hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Volcamos la mascara de la banda a una mascara del tamano completo
        mask_full[y1:y2, x1:x2] = mask_clean

        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = max(contours, key=cv2.contourArea) if contours else None
        if best_cnt is not None and cv2.contourArea(best_cnt) > 50:
            bx, by, bw, bh = cv2.boundingRect(best_cnt)
            bbox_abs = (x1 + bx, y1 + by, bw, bh)

    return mask_full, bbox_abs, (x1, y1, x2, y2)


def capturar_en_vivo(forzar_w=800, forzar_h=600):
    """Captura un fotograma de la ventana del juego (mismo metodo que el env).

    La resolucion del juego es siempre 800x600. Si el escalado de Windows no
    esta al 100%, GetClientRect/ClientToScreen devuelven coordenadas logicas
    mientras que mss captura fisicas, y la region sale desplazada. Para evitarlo
    activamos DPI-awareness y forzamos el tamano del area de cliente a 800x600.
    """
    import mss
    import pygetwindow as gw
    import ctypes
    from ctypes import wintypes

    # Coordenadas en pixeles fisicos (imprescindible con escalado != 100%)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    wins = gw.getWindowsWithTitle(GAME_TITLE)
    if not wins:
        raise RuntimeError(f"No se encontro la ventana '{GAME_TITLE}'. Abre el juego.")
    win = wins[0]
    hwnd = win._hWnd
    win.activate()

    # Origen del area de cliente en coordenadas de pantalla
    point = wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))

    rect = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
    w_real, h_real = rect.right - rect.left, rect.bottom - rect.top

    # Forzamos 800x600 salvo que el cliente real ya sea exactamente eso
    w = forzar_w if forzar_w else w_real
    h = forzar_h if forzar_h else h_real
    if (w_real, h_real) != (w, h):
        print(f"[INFO] Cliente medido {w_real}x{h_real}; capturando {w}x{h} (forzado).")

    monitor = {"top": point.y, "left": point.x, "width": w, "height": h}

    with mss.mss() as sct:
        img = np.array(sct.grab(monitor))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def generar_figura(frame_bgr, mask_full, bbox_abs, banda, salida):
    x1, y1, x2, y2 = banda
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Panel 1: original con la banda de busqueda marcada
    panel_orig = frame_rgb.copy()
    cv2.rectangle(panel_orig, (x1, y1), (x2, y2), (255, 200, 0), 2)

    # Panel 2: mascara verde (en escala de grises -> visible como blanco)
    panel_mask = mask_full

    # Panel 3: original + contorno principal y bounding box
    panel_cnt = frame_rgb.copy()
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel_cnt, contours, -1, (255, 0, 0), 2)
    if bbox_abs is not None:
        bx, by, bw, bh = bbox_abs
        cv2.rectangle(panel_cnt, (bx, by), (bx + bw, by + bh), (255, 0, 255), 2)
        cy = by + bh // 2
        cv2.line(panel_cnt, (bx, cy), (bx + bw, cy), (0, 255, 255), 2)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    axs[0].imshow(panel_orig)
    axs[0].set_title("1. Fotograma original\n(banda de busqueda en amarillo)")
    axs[1].imshow(panel_mask, cmap="gray")
    axs[1].set_title("2. Mascara verde HSV\n(tras morfologia)")
    axs[2].imshow(panel_cnt)
    axs[2].set_title("3. Contorno principal\n+ centro detectado")
    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura guardada en: {salida}")
    print(f">>> SUBIR: {salida} -> imaxes/{os.path.basename(salida)}")


def main():
    ap = argparse.ArgumentParser(description="Genera la figura de deteccion del jugador.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--imagen", help="Ruta a una captura/fotograma ya guardado.")
    src.add_argument("--live", action="store_true", help="Capturar de la ventana del juego.")
    ap.add_argument("--salida", default="metrics/deteccion_jugador.png", help="Ruta de salida.")
    ap.add_argument("--tol", type=int, default=12,
                    help="Tolerancia HSV +-tol (0 = rango exacto del proyecto).")
    ap.add_argument("--no-forzar", action="store_true",
                    help="No forzar 800x600; usar el tamano real del cliente.")
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

    mask_full, bbox_abs, banda = detectar(frame, tol=args.tol)
    if bbox_abs is None:
        print("[AVISO] No se detecto contorno. Prueba a subir --tol (p.ej. --tol 25).")
    generar_figura(frame, mask_full, bbox_abs, banda, args.salida)


if __name__ == "__main__":
    main()
