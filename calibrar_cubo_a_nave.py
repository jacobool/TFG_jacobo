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
PLAYER_BAND_W = 0.08  
LOWER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([45, 255, 255], dtype=np.uint8)

# Área mínima para considerar que el piloto está "VIVO"
GREEN_AREA_MIN = 130  

# --- EL CEREBRO DEL INTERRUPTOR: Basado en TAMAÑO ---
# Si la pieza verde es más grande que esto, asumimos que es un CUBO macizo.
# Si es más pequeña (como tu 171), asumimos que es la cabina de la NAVE.
# Basándonos en tu captura (171), ponemos el corte en 220.
UMBRAL_AREA_SPLIT = 220 

def get_window_rect(hwnd):
    rect = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    point = wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
    return {"top": point.y, "left": point.x, "width": w, "height": h}

def detectar_modo_y_jugador(frame_bgr):
    h, w = frame_bgr.shape[:2]
    player_x = int(w * PLAYER_X_REL)
    band_half = int(w * PLAYER_BAND_W / 2)
    x1 = max(0, player_x - band_half)
    x2 = min(w, player_x + band_half)
    y1, y2 = int(h * 0.08), int(h * 0.92)
    band_bgr = frame_bgr[y1:y2, x1:x2]
    
    modo_detectado = "DESCONOCIDO"
    area_actual = 0.0
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
            # 1. Medimos el área de la pieza verde
            largest_piece_area = cv2.contourArea(best_cnt)
            area_actual = largest_piece_area
            
            # 2. Verificamos que esté "vivo" (más de 130)
            if largest_piece_area >= GREEN_AREA_MIN:
                
                # 3. LÓGICA DEL INTERRUPTOR DE MODO
                if largest_piece_area >= UMBRAL_AREA_SPLIT:
                    # Es una masa verde grande y sólida -> CUBO
                    modo_detectado = "CUBO"
                else:
                    # Es una masa verde pequeña (como tu 171) -> NAVE
                    modo_detectado = "NAVE"
                
                bx, by, bw, bh = cv2.boundingRect(best_cnt)
                bbox_abs = (x1 + bx, y1 + by, bw, bh)
                
    return modo_detectado, area_actual, bbox_abs

# --- BUCLE PRINCIPAL DE CALIBRACIÓN ---
windows = gw.getWindowsWithTitle(GAME_TITLE)
if not windows:
    print("❌ Ábrelo primero.")
    exit()

win = windows[0]
hwnd = win._hWnd
win.activate()
sct = mss.mss()

print("✅ Calibrador del DIRECTOR (Interruptor de Área) iniciado.")
print("Juega y observa cómo el texto cambia al ver la diferencia de tamaño.")

while True:
    monitor = get_window_rect(hwnd)
    img = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h_f, w_f = frame.shape[:2]

    modo, area_actual, bbox_abs = detectar_modo_y_jugador(frame)

    # --- DIBUJO DE INTERFAZ DE CALIBRACIÓN ---
    
    # Hitbox verde para confirmar detección (PARA TI)
    if bbox_abs is not None:
        cv2.rectangle(frame, (bbox_abs[0], bbox_abs[1]), 
                             (bbox_abs[0] + bbox_abs[2], bbox_abs[1] + bbox_abs[3]), (0, 255, 0), 2)
                             
    # Colores molones según el modo
    if modo == "NAVE":
        color_modo = (255, 0, 255) # Magenta
    elif modo == "CUBO":
        color_modo = (255, 255, 0) # Cyan
    else:
        color_modo = (0, 0, 255)   # Rojo (Muerto/Invisible)

    # Textos en pantalla
    cv2.putText(frame, f"MODO: {modo}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, color_modo, 3)
    cv2.putText(frame, f"Area Actual: {area_actual}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Umbral Corte Area: {UMBRAL_AREA_SPLIT}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Calibrador del Director - Test de Área", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
sct.close()