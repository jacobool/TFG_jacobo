import mss
import pygetwindow as gw
import numpy as np
import cv2
import time

# ----------------------------
# CONFIG
# ----------------------------

GAME_TITLE = "Geometry Dash"
CHECK_INTERVAL = 0.05          # segundos entre capturas
GREEN_THRESHOLD = 50           # píxeles verdes mínimos para considerar "vivo"
DEATH_CONFIRM_FRAMES = 3       # frames CONSECUTIVOS bajo umbral para confirmar muerte
DEATH_COOLDOWN = 1.0           # segundos de cooldown tras detectar muerte
DOWNSCALE = 0.5                # factor de reducción (0.5 = mitad de resolución)
FPS_REPORT_INTERVAL = 5.0     # cada cuántos segundos mostrar FPS
DEBUG_VIEW = True              # mostrar ventana con la máscara verde en tiempo real

# ROI relativa a la ventana del juego (dónde suele estar el cubo)
# (x_start%, y_start%, x_end%, y_end%)  —  ajustar según el nivel
ROI_REL = (0.05, 0.30, 0.95, 0.95)

# Rango HSV del verde del cubo (constantes fuera del loop)
LOWER_GREEN = np.array([40, 150, 150], dtype=np.uint8)
UPPER_GREEN = np.array([80, 255, 255], dtype=np.uint8)

# ----------------------------
# OBTENER VENTANA
# ----------------------------

windows = gw.getWindowsWithTitle(GAME_TITLE)

if not windows:
    print("No se encontró la ventana del juego.")
    exit()

window = windows[0]

print(f"Ventana encontrada: pos=({window.left}, {window.top}) "
      f"size=({window.width}x{window.height})")


def compute_monitor(win):
    """Calcula la región de captura (ROI) a partir de la ventana."""
    rx0, ry0, rx1, ry1 = ROI_REL
    left = win.left + int(win.width * rx0)
    top = win.top + int(win.height * ry0)
    width = int(win.width * (rx1 - rx0))
    height = int(win.height * (ry1 - ry0))
    return {"left": left, "top": top, "width": width+10, "height": height}


# Cachear monitor — se recalcula solo si la ventana se mueve/redimensiona
_last_win_rect = (window.left, window.top, window.width+10, window.height)
monitor = compute_monitor(window)

time.sleep(2)
print("Iniciando detección de muerte...")

# ----------------------------
# LOOP PRINCIPAL
# ----------------------------

is_dead = False
last_death_time = 0.0
low_streak = 0                 # frames consecutivos con pocos verdes
frame_count = 0
fps_timer = time.perf_counter()

with mss.mss() as sct:
    while True:
        t_start = time.perf_counter()

        # --- Actualizar monitor si la ventana se movió ---
        cur_rect = (window.left, window.top, window.width, window.height)
        if cur_rect != _last_win_rect:
            monitor = compute_monitor(window)
            _last_win_rect = cur_rect

        # --- Captura ---
        screenshot = sct.grab(monitor)
        # np.asarray evita copia innecesaria del buffer de mss
        frame = np.asarray(screenshot)

        # --- Quitar canal alfa con slicing (más rápido que cvtColor BGRA→BGR) ---
        bgr = frame[:, :, :3]

        # --- Downscale para reducir coste de cvtColor + inRange ---
        if DOWNSCALE < 1.0:
            bgr = cv2.resize(bgr, None, fx=DOWNSCALE, fy=DOWNSCALE,
                             interpolation=cv2.INTER_NEAREST)

        # --- HSV + máscara verde ---
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        green_pixels = cv2.countNonZero(mask)

        # --- Debug: mostrar máscara y conteo en ventana ---
        if DEBUG_VIEW:
            debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            status = "MUERTO" if is_dead else "VIVO"
            color = (0, 0, 255) if is_dead else (0, 255, 0)
            cv2.putText(debug, f"Verdes: {green_pixels}  [{status}]",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(debug, f"Streak: {low_streak}/{DEATH_CONFIRM_FRAMES}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.imshow("Debug - Mascara Verde", debug)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # --- Máquina de estados: vivo ↔ muerto (con confirmación) ---
        now = time.perf_counter()

        if green_pixels < GREEN_THRESHOLD:
            low_streak += 1
        else:
            low_streak = 0

        if not is_dead and low_streak >= DEATH_CONFIRM_FRAMES:
            is_dead = True
            last_death_time = now
            print(f"💀 MUERTE DETECTADA  (verdes={green_pixels}, "
                  f"streak={low_streak})")

        elif is_dead and green_pixels >= GREEN_THRESHOLD:
            # Cubo visible de nuevo → resurrección
            if now - last_death_time > DEATH_COOLDOWN:
                is_dead = False
                low_streak = 0
                print(f"✅ RESURRECCIÓN  (verdes={green_pixels})")

        # --- Rate limiting preciso ---
        elapsed = time.perf_counter() - t_start
        remaining = CHECK_INTERVAL - elapsed
        if remaining > 0:
            time.sleep(remaining)

if DEBUG_VIEW:
    cv2.destroyAllWindows()