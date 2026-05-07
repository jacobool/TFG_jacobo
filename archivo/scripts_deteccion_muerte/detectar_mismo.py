import cv2
import numpy as np
import mss
import pygetwindow as gw
import ctypes
from ctypes import wintypes
import time

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

GAME_TITLE = "Geometry Dash"

PLAYER_X_REL  = 0.37
PLAYER_BAND_W = 0.06

# Rango HSV del borde verde (ajustado con tu pixel HSV=[46,255,255])
LOWER_GREEN = np.array([46, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([46, 255, 255], dtype=np.uint8)

DEATH_TIMEOUT = 0.15  # segundos para considerar muerte tras perder jugador


def get_window_rect(hwnd):
    rect = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    point = wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
    return {"top": point.y, "left": point.x, "width": w, "height": h}


def detect_player_and_band(frame_bgr):
    """Devuelve display, player_center_y, green_area"""
    h, w = frame_bgr.shape[:2]
    display = frame_bgr.copy()

    player_x   = int(w * PLAYER_X_REL)
    band_half  = int(w * PLAYER_BAND_W / 2)
    x1 = max(0, player_x - band_half)
    x2 = min(w, player_x + band_half)
    y1, y2 = int(h * 0.08), int(h * 0.92)

    band_bgr = frame_bgr[y1:y2, x1:x2]
    player_center_y = None
    green_area = 0

    if band_bgr.size > 0:
        band_hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(band_hsv, LOWER_GREEN, UPPER_GREEN)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        green_area = cv2.countNonZero(mask_clean)

        # Contorno principal para dibujar
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = max(contours, key=cv2.contourArea) if contours else None
        
        if best_cnt is not None and cv2.contourArea(best_cnt) > 50:
            bx, by, bw, bh = cv2.boundingRect(best_cnt)
            abs_cx = x1 + bx + bw // 2
            abs_cy = y1 + by + bh // 2
            side = int(w * PLAYER_BAND_W)
            sq_x1 = abs_cx - side // 2
            sq_y1 = abs_cy - side // 2
            sq_x2 = abs_cx + side // 2
            sq_y2 = abs_cy + side // 2

            player_center_y = abs_cy
            cv2.rectangle(display, (sq_x1, sq_y1), (sq_x2, sq_y2),
                          (0, 255, 0), 2)
            cv2.drawMarker(display, (abs_cx, abs_cy), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)

    # Banda de debug
    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 100, 0), 1)

    return display, player_center_y, green_area


def main():
    windows = gw.getWindowsWithTitle(GAME_TITLE)
    if not windows:
        print(f"No se encontró ventana con título '{GAME_TITLE}'")
        return

    win = windows[0]
    hwnd = win._hWnd
    win.activate()

    # Estado del intento
    attempt = 1
    player_last_seen = 0.0
    alive = False

    with mss.mss() as sct:
        while True:
            monitor = get_window_rect(hwnd)
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            now = time.time()

            display, player_y, green_area = detect_player_and_band(frame)

            # Jugador VISIBLE si hay suficiente verde
            player_visible = green_area > 100

            if player_visible:
                player_last_seen = now
                if not alive:  # Primer frame del respawn
                    alive = True
                    attempt += 1
                    print(f"RESPAWN → Attempt {attempt}")
            else:
                # Jugador NO VISIBLE
                if alive and (now - player_last_seen) >= DEATH_TIMEOUT:
                    # Timeout alcanzado → MUERTE
                    print(f"MUERTE → Attempt {attempt} terminado")
                    alive = False

            # Overlays informativos
            cv2.putText(display, f"Attempt {attempt}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(display, f"Green: {green_area:.0f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            status_text = "ALIVE" if alive else "DEAD"
            cv2.putText(display, status_text, (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("GD TRACKER", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
