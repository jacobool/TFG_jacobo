import mss
import pygetwindow as gw
import numpy as np
import cv2
import time

# ----------------------------
# CONFIG
# ----------------------------

GAME_TITLE = "Geometry Dash"
TEMPLATE_PATH = "attempt_template.png"
THRESHOLD = 0.7
CHECK_INTERVAL = 0.01          # segundos entre capturas
DEATH_COOLDOWN = 0.5         # cooldown tras detectar muerte
DEBUG_VIEW = True              # mostrar ventana debug con la región analizada

# Multi-escala: rango de escalas para buscar el template
# (cubre ventanas desde ~50% hasta ~200% del tamaño con el que hiciste el template)
SCALES = np.linspace(0.4, 2.0, 20)

# ROI relativa: zona superior donde aparece "Attempt X"
ROI_REL = (0.3, 0.18, 0.7, 0.65)  # (x0%, y0%, x1%, y1%)

# ----------------------------
# Cargar template
# ----------------------------
template_full = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
if template_full is None:
    print(f"ERROR: No se pudo cargar '{TEMPLATE_PATH}'")
    exit()

t_h, t_w = template_full.shape[:2]
print(f"Template cargado: {t_w}x{t_h} px")

# ----------------------------
# Obtener ventana
# ----------------------------
windows = gw.getWindowsWithTitle(GAME_TITLE)

if not windows:
    print("No se encontró la ventana del juego.")
    exit()

window = windows[0]
print(f"Ventana detectada: pos=({window.left},{window.top}) "
      f"size=({window.width}x{window.height})")


def compute_monitor(win):
    """Calcula la región de captura a partir de ROI_REL."""
    x0, y0, x1, y1 = ROI_REL
    left = win.left + int(win.width * x0)
    top = win.top + int(win.height * y0)
    width = int(win.width * (x1 - x0))
    height = int(win.height * (y1 - y0))
    return {"left": left, "top": top, "width": width, "height": height}


def multiscale_match(gray, template, scales, threshold):
    """
    Busca el template a múltiples escalas.
    Devuelve (best_score, best_loc, best_scale, best_tw, best_th) o None.
    """
    best_score = -1
    best_loc = None
    best_scale = 1.0
    best_tw, best_th = t_w, t_h
    gh, gw_ = gray.shape[:2]

    for scale in scales:
        new_w = int(t_w * scale)
        new_h = int(t_h * scale)

        # Saltar si el template reescalado es más grande que la imagen
        if new_w >= gw_ or new_h >= gh or new_w < 10 or new_h < 10:
            continue

        resized = cv2.resize(template, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_scale = scale
            best_tw, best_th = new_w, new_h

    if best_score >= threshold:
        return best_score, best_loc, best_scale, best_tw, best_th
    return None


# Cachear monitor
_last_win_rect = (window.left, window.top, window.width, window.height)
monitor = compute_monitor(window)

time.sleep(2)
print("Iniciando detección de muerte...")

# ----------------------------
# Loop principal
# ----------------------------

is_dead = False
last_death_time = 0.0

with mss.mss() as sct:
    while True:
        t_start = time.perf_counter()

        # --- Captura ---
        screenshot = sct.grab(monitor)
        frame = np.asarray(screenshot)

        # --- Convertir a gris ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        # --- Template matching multi-escala ---
        match = multiscale_match(gray, template_full, SCALES, THRESHOLD)

        now = time.perf_counter()

        if match is not None:
            score, loc, scale, tw, th = match

            if not is_dead:
                is_dead = True
                last_death_time = now
                print(f"💀 MUERTE DETECTADA  (score={score:.2f}, "
                      f"scale={scale:.2f})")

        else:
            # No se encontró "Attempt" → jugador vivo
            if is_dead and (now - last_death_time > DEATH_COOLDOWN):
                is_dead = False
                print("✅ RESURRECCIÓN")

        # --- Debug: mostrar ventana con región analizada ---
        if DEBUG_VIEW:
            debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if match is not None:
                score, loc, scale, tw, th = match
                # Rectángulo verde en la detección
                top_left = loc
                bottom_right = (loc[0] + tw, loc[1] + th)
                cv2.rectangle(debug, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(debug,
                            f"score={score:.2f} scale={scale:.2f}",
                            (loc[0], loc[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Estado y info de la ROI
            status = "MUERTO" if is_dead else "VIVO"
            color = (0, 0, 255) if is_dead else (0, 255, 0)
            cv2.putText(debug, f"[{status}]  ROI: {monitor['width']}x{monitor['height']}",
                        (10, debug.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow("Debug - Region Analizada", debug)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # --- Rate limiting ---
        elapsed = time.perf_counter() - t_start
        remaining = CHECK_INTERVAL - elapsed
        if remaining > 0:
            time.sleep(remaining)

if DEBUG_VIEW:
    cv2.destroyAllWindows()