import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

import cv2
import mss
import numpy as np
import pygetwindow as gw
import ctypes
from ctypes import wintypes
import time  # <-- IMPORTANTE: Añadimos la librería de tiempo

GAME_TITLE = "Geometry Dash"

# Configuramos la velocidad deseada
TARGET_FPS = 15
STEP_DURATION = 1.0 / TARGET_FPS

# Tu misma función para encontrar la ventana
def get_window_rect(hwnd):
    rect = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    point = wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
    
    # Calculamos el recorte ANTES de capturar
    margen_izq = int(w * 0.20)
    margen_der = int(w * 0.80)
    ancho_real = margen_der - margen_izq
    
    # Le decimos a MSS que empiece a capturar más a la derecha, 
    # y que el ancho total sea solo ese 60% central.
    return {
        "top": point.y, 
        "left": point.x + margen_izq, 
        "width": ancho_real, 
        "height": h
    }
windows = gw.getWindowsWithTitle(GAME_TITLE)
if not windows:
    print("❌ Abre Geometry Dash primero.")
    exit()

hwnd = windows[0]._hWnd
sct = mss.mss()

print(f"👀 Mostrando la visión EXACTA de la IA a {TARGET_FPS} FPS... (Presiona 'Q' para salir)")

while True:
    step_start = time.perf_counter()  # <-- 1. Iniciamos el cronómetro del frame

    # 1. Capturamos la pantalla
    monitor = get_window_rect(hwnd)
    img = np.array(sct.grab(monitor))
    cropped_frame = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    blurred = cv2.blur(cropped_frame, (3, 3))
    edges = cv2.Canny(blurred, threshold1=170, threshold2=270)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    #gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    #_, mask = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    
    # 3. Desenfoque y Rayos X
    #blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    #edges = cv2.Canny(mask, threshold1=50, threshold2=255)
    
    # 4. LA COMPRESIÓN A 84x84
    ai_vision = cv2.resize(dilated, (84, 84))
    
    # 5. AMPLIACIÓN PARA TUS OJOS
    display_vision = cv2.resize(ai_vision, (420, 420), interpolation=cv2.INTER_NEAREST)
    
    # 6. Mostramos el resultado
    cv2.imshow("Vision IA - Geometry Dash (84x84 a 15 FPS)", display_vision)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 7. EL LIMITADOR DE TIEMPO (Igual que en tu IA)
    elapsed = time.perf_counter() - step_start
    remaining = STEP_DURATION - elapsed
    if remaining > 0:
        time.sleep(remaining)  # <-- Obligamos al código a esperar para clavar los 15 FPS

cv2.destroyAllWindows()
sct.close()