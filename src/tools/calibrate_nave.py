import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

import cv2
import numpy as np
import mss
import pygetwindow as gw
import ctypes
from ctypes import wintypes
import time

# --- 1. CONFIGURACIÓN ---
GAME_TITLE = "Geometry Dash"
PLAYER_X_REL = 0.345
PLAYER_BAND_W = 0.08  # Tu ajuste perfecto
LOWER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
GREEN_AREA_MIN = 130  

# --- TAMAÑO ARTIFICIAL DE LA NAVE ---
SHIP_HITBOX_WIDTH = 55   # Un poco menos ancho para que no sobre a la derecha
SHIP_HITBOX_HEIGHT = 45  # Alto para cubrir todo
SHIP_OFFSET_X = 0        # Centrado horizontalmente con la cabina
SHIP_OFFSET_Y = 10       # NUEVO: Empuja la caja 10 píxeles hacia ABAJO para cubrir la panza

def get_window_rect(hwnd):
    rect = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    point = wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
    return {"top": point.y, "left": point.x, "width": w, "height": h}

# --- 2. DETECCIÓN DE JUGADOR (Con Hitbox Agrandada) ---
def detect_player_and_band(frame_bgr):
    h, w = frame_bgr.shape[:2]
    player_x = int(w * PLAYER_X_REL)
    band_half = int(w * PLAYER_BAND_W / 2)
    x1 = max(0, player_x - band_half)
    x2 = min(w, player_x + band_half)
    y1, y2 = int(h * 0.08), int(h * 0.92)
    band_bgr = frame_bgr[y1:y2, x1:x2]
    
    player_center_y = h / 2
    green_area = 0
    bbox_abs = None  
    
    if band_bgr.size > 0:
        band_hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(band_hsv, LOWER_GREEN, UPPER_GREEN)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = max(contours, key=cv2.contourArea) if contours else None
        
        if best_cnt is not None:
            largest_piece_area = cv2.contourArea(best_cnt)
            if largest_piece_area >= GREEN_AREA_MIN:
                # Centro real del cristal del piloto
                bx, by, bw, bh = cv2.boundingRect(best_cnt)
                center_x_local = bx + bw // 2
                center_y_local = by + bh // 2
                
                # Creamos la caja artificial y le aplicamos los offsets X e Y
                expanded_x = center_x_local - (SHIP_HITBOX_WIDTH // 2) + SHIP_OFFSET_X
                expanded_y = center_y_local - (SHIP_HITBOX_HEIGHT // 2) + SHIP_OFFSET_Y
                
                # Evitar salirnos de la pantalla
                expanded_x = max(0, expanded_x)
                expanded_y = max(0, expanded_y)
                
                abs_cy = y1 + center_y_local + SHIP_OFFSET_Y # El centro Y que pasamos también baja
                player_center_y = abs_cy
                
                bbox_abs = (x1 + expanded_x, y1 + expanded_y, SHIP_HITBOX_WIDTH, SHIP_HITBOX_HEIGHT)
                green_area = largest_piece_area
            else:
                green_area = 0
                bbox_abs = None
                
    return player_center_y, green_area, bbox_abs

# --- 3. BUCLE PRINCIPAL DE CALIBRACIÓN ---
windows = gw.getWindowsWithTitle(GAME_TITLE)
if not windows:
    print("❌ No se encontró la ventana de Geometry Dash. Ábrelo primero.")
    exit()

win = windows[0]
hwnd = win._hWnd
win.activate()

sct = mss.mss()

print("✅ Calibrador de la NAVE iniciado.")
print("Pulsa 'q' en la ventana para salir.")

while True:
    t_start = time.time()
    
    monitor = get_window_rect(hwnd)
    img = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = frame.shape[:2]

    # 1. Detección del jugador
    player_center_y, green_area, bbox_abs = detect_player_and_band(frame)
    is_alive = green_area >= GREEN_AREA_MIN

    # =======================================================
    # VISIÓN DE LA IA (Tu versión intacta)
    # =======================================================
    x1_ai, x2_ai = int(w * 0.20), int(w * 0.90)
    frame_color_cropped = frame[:, x1_ai:x2_ai]
    
    lower_white = np.array([220, 220, 220], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask_white = cv2.inRange(frame_color_cropped, lower_white, upper_white)
    
    kernel_ai = np.ones((2, 2), np.uint8)
    vision_clean = cv2.dilate(mask_white, kernel_ai, iterations=1)

    # Dibuja la hitbox gigante en la IA
    if is_alive and bbox_abs is not None:
        x_abs, y_abs, bw, bh = bbox_abs
        p_x1 = x_abs - x1_ai
        p_y1 = y_abs
        if p_x1 >= 0:
            cv2.rectangle(vision_clean, (p_x1, p_y1), (p_x1 + bw, p_y1 + bh), 255, -1)

    ai_vision_84 = cv2.resize(vision_clean, (84, 84), interpolation=cv2.INTER_AREA)

    # --- DIBUJADO DE LA INTERFAZ ---
    p_x_center = int(w * PLAYER_X_REL)
    band_half = int(w * PLAYER_BAND_W / 2)
    # Las líneas amarillas ahora estarán mucho más separadas
    cv2.rectangle(frame, (max(0, p_x_center - band_half), int(h * 0.08)), 
                         (min(w, p_x_center + band_half), int(h * 0.92)), (0, 255, 255), 1)

    if is_alive and bbox_abs is not None:
        # Aquí verás el recuadro verde gigante cubriendo toda la nave
        cv2.rectangle(frame, (bbox_abs[0], bbox_abs[1]), 
                             (bbox_abs[0] + bbox_abs[2], bbox_abs[1] + bbox_abs[3]), (0, 255, 0), 2)
        cv2.circle(frame, (int(p_x_center), int(player_center_y)), 4, (0, 0, 255), -1)

    status_text = "VIVO" if is_alive else "MUERTO"
    color_text = (0, 255, 0) if is_alive else (0, 0, 255)
    cv2.putText(frame, f"Estado: {status_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_text, 2)
    cv2.putText(frame, f"Area Verde: {green_area} (Min: {GREEN_AREA_MIN})", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    ai_vision_bgr = cv2.cvtColor(ai_vision_84, cv2.COLOR_GRAY2BGR)
    ai_display_size = 252 
    ai_vision_display = cv2.resize(ai_vision_bgr, (ai_display_size, ai_display_size), interpolation=cv2.INTER_NEAREST)
    
    frame[h-ai_display_size:h, w-ai_display_size:w] = ai_vision_display
    cv2.rectangle(frame, (w-ai_display_size, h-ai_display_size), (w, h), (255, 0, 255), 2)
    cv2.putText(frame, "Vision IA (Nave)", (w-ai_display_size + 5, h-ai_display_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    t_end = time.time()
    fps = 1.0 / (t_end - t_start + 0.0001)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Calibrador IA - Nave", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
sct.close()