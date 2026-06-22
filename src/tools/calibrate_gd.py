import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

import numpy as np
import cv2
import mss
import pygetwindow as gw
import ctypes
from ctypes import wintypes
import time

# --- CONSTANTES EXACTAS DE TU CÓDIGO ---
GAME_TITLE = "Geometry Dash"
PLAYER_X_REL = 0.345
PLAYER_BAND_W = 0.065
LOWER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
GREEN_AREA_MIN = 100

def get_window_rect(hwnd):
    rect = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    point = wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
    return {"top": point.y, "left": point.x, "width": w, "height": h}

def main():
    windows = gw.getWindowsWithTitle(GAME_TITLE)
    if not windows:
        print("❌ No se encontró la ventana de Geometry Dash.")
        return
    
    win = windows[0]
    hwnd = win._hWnd
    sct = mss.mss()
    
    print("✅ Calibrador iniciado. Presiona 'Q' en la ventana para salir.")

    while True:
        # 1. Capturar pantalla
        monitor = get_window_rect(hwnd)
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = frame.shape[:2]

        # =======================================================
        # A. LÓGICA DE DETECCIÓN DEL JUGADOR (El Filtro Verde)
        # =======================================================
        player_x = int(w * PLAYER_X_REL)
        band_half = int(w * PLAYER_BAND_W / 2)
        x1 = max(0, player_x - band_half)
        x2 = min(w, player_x + band_half)
        y1, y2 = int(h * 0.08), int(h * 0.92)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "Zona Busqueda", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        band_bgr = frame[y1:y2, x1:x2]
        green_area = 0
        is_alive = False

        if band_bgr.size > 0:
            band_hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
            mask_green = cv2.inRange(band_hsv, LOWER_GREEN, UPPER_GREEN)
            kernel_green = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel_green, iterations=1)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_green, iterations=1)
            
            green_area = cv2.countNonZero(mask_clean)

            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_cnt = max(contours, key=cv2.contourArea) if contours else None

            if best_cnt is not None and cv2.contourArea(best_cnt) > 50:
                bx, by, bw, bh = cv2.boundingRect(best_cnt)
                abs_x1, abs_y1 = x1 + bx, y1 + by
                abs_x2, abs_y2 = abs_x1 + bw, abs_y1 + bh
                
                cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
                cv2.circle(frame, (abs_x1 + bw//2, abs_y1 + bh//2), 5, (0, 0, 255), -1)

            if green_area >= GREEN_AREA_MIN:
                is_alive = True

            # PIP Superior Derecho (Máscara Verde)
            mask_bgr = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
            pip_h, pip_w = mask_bgr.shape[:2]
            frame[0:pip_h, w-pip_w:w] = mask_bgr
            cv2.rectangle(frame, (w-pip_w, 0), (w, pip_h), (0, 255, 255), 2)
            cv2.putText(frame, "Lo que ve el detector", (w-pip_w + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

# =======================================================
        # =======================================================
        # B. LÓGICA DE VISIÓN DE LA IA (Visión de Hitboxes Pura)
        # =======================================================
        x1_ai, x2_ai = int(w * 0.20), int(w * 0.90)
        frame_gray_cropped = cv2.cvtColor(img[:, x1_ai:x2_ai], cv2.COLOR_BGRA2GRAY)
        
        # 1. Umbral MUY ALTO (205). Mata fondos oscuros/medios y deja el blanco.
        _, thresh_full = cv2.threshold(frame_gray_cropped, 205, 255, cv2.THRESH_BINARY)
        
        # 2. Engrosamos un poco con 2x2 (ideal para tu resolución de 800x600)
        kernel_ai = np.ones((2, 2), np.uint8)
        vision_clean = cv2.dilate(thresh_full, kernel_ai, iterations=1)

        # 3. EL TRUCO: Dibujamos nosotros mismos al jugador artificialmente
        if is_alive and best_cnt is not None:
            # Calculamos las coordenadas del jugador relativas al recorte de la IA
            p_x1 = (x1 + bx) - x1_ai
            p_y1 = y1 + by
            p_x2 = p_x1 + bw
            p_y2 = p_y1 + bh
            
            # Dibujamos un bloque blanco sólido exactamente donde está el jugador
            cv2.rectangle(vision_clean, (p_x1, p_y1), (p_x2, p_y2), 255, -1)

        # 4. LA SOLUCIÓN AL SUELO: Añadimos interpolation=cv2.INTER_AREA
        # Esto hace que las líneas finas no se evaporen al comprimir la imagen
        ai_vision_84 = cv2.resize(vision_clean, (84, 84), interpolation=cv2.INTER_AREA)

        # --- Ampliamos para ver en la pantalla del calibrador ---
        ai_vision_bgr = cv2.cvtColor(ai_vision_84, cv2.COLOR_GRAY2BGR)
        ai_display_size = 252 
        ai_vision_display = cv2.resize(ai_vision_bgr, (ai_display_size, ai_display_size), interpolation=cv2.INTER_NEAREST)
        
        # Colocamos la visión
        frame[h-ai_display_size:h, w-ai_display_size:w] = ai_vision_display
        cv2.rectangle(frame, (w-ai_display_size, h-ai_display_size), (w, h), (255, 0, 255), 2)
        cv2.putText(frame, "Vision IA (Hitbox Pura)", (w-ai_display_size + 5, h-ai_display_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        # C. TEXTOS INFORMATIVOS PRINCIPALES
        # =======================================================
        status_text = "VIVO" if is_alive else "MUERTO"
        color = (0, 255, 0) if is_alive else (0, 0, 255)
        
        cv2.putText(frame, f"Estado: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"Area Verde: {green_area} / Min: {GREEN_AREA_MIN}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Mostrar la ventana final
        cv2.imshow("Calibrador Geometry Dash", frame)

        # Presionar 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    sct.close()

if __name__ == "__main__":
    main()