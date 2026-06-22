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
import torch
from torchvision import models
import torchvision.transforms as T

# --- 1. CONFIGURACIÓN BÁSICA ---
GAME_TITLE = "Geometry Dash"
PLAYER_X_REL = 0.345
PLAYER_BAND_W = 0.08  
LOWER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
GREEN_AREA_MIN = 130  
UMBRAL_AREA_SPLIT = 220 

# --- 2. CARGA DE LA RED NEURONAL DE SEGMENTACIÓN ---
print("⏳ Descargando/Cargando modelo DeepLabV3 (PyTorch)... Ten paciencia.")
# Usamos MobileNet porque es la más ligera para CPU
segmentation_model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
segmentation_model.eval() 
device = torch.device('cpu') 

# Transformaciones matemáticas obligatorias para la red
preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("✅ Red Neuronal lista.")

def get_window_rect(hwnd):
    rect = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    point = wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
    return {"top": point.y, "left": point.x, "width": w, "height": h}

def detectar_jugador(frame_bgr):
    # (Mantenemos OpenCV solo para encontrar la caja verde del jugador)
    h, w = frame_bgr.shape[:2]
    player_x = int(w * PLAYER_X_REL)
    band_half = int(w * PLAYER_BAND_W / 2)
    x1 = max(0, player_x - band_half)
    x2 = min(w, player_x + band_half)
    y1, y2 = int(h * 0.08), int(h * 0.92)
    band_bgr = frame_bgr[y1:y2, x1:x2]
    
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
            area = cv2.contourArea(best_cnt)
            if area >= GREEN_AREA_MIN:
                bx, by, bw, bh = cv2.boundingRect(best_cnt)
                
                # Expandimos si es nave
                if area < UMBRAL_AREA_SPLIT: 
                    cx = bx + bw // 2
                    cy = by + bh // 2
                    ex = max(0, cx - 27)
                    ey = max(0, cy - 22 + 10)
                    bbox_abs = (x1 + ex, y1 + ey, 55, 45)
                else:
                    bbox_abs = (x1 + bx, y1 + by, bw, bh)
                
    return bbox_abs

# --- BUCLE PRINCIPAL ---
windows = gw.getWindowsWithTitle(GAME_TITLE)
if not windows:
    print("❌ Abre Geometry Dash primero.")
    exit()

win = windows[0]
hwnd = win._hWnd
win.activate()
sct = mss.mss()

print("🚀 Calibrador Deep Learning iniciado. ¡Mira los FPS!")

while True:
    t_start = time.perf_counter()
    
    monitor = get_window_rect(hwnd)
    img = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h_f, w_f = frame.shape[:2]

    bbox_abs = detectar_jugador(frame)

    # --- VISIÓN DE LA IA MEDIANTE RED NEURONAL ---
    x1_ai, x2_ai = int(w_f * 0.20), int(w_f * 0.90)
    frame_color_cropped = frame[:, x1_ai:x2_ai]
    
    # 1. Preprocesar para PyTorch
    frame_rgb = cv2.cvtColor(frame_color_cropped, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(frame_rgb).unsqueeze(0).to(device)
    
    # 2. Inferencia (Puro cálculo de CPU)
    with torch.no_grad():
        output = segmentation_model(input_tensor)['out'][0]
    
    # 3. Postprocesamiento (Todo lo que no sea fondo clase 0, lo ponemos blanco)
    output_predictions = output.argmax(0).byte().numpy()
    vision_nn = np.where(output_predictions > 0, 255, 0).astype(np.uint8)

    # Añadimos la hitbox del jugador en blanco
    if bbox_abs is not None:
        x_abs, y_abs, bw, bh = bbox_abs
        p_x1 = x_abs - x1_ai
        if p_x1 >= 0:
            cv2.rectangle(vision_nn, (p_x1, y_abs), (p_x1 + bw, y_abs + bh), 255, -1)
    
    ai_vision_84 = cv2.resize(vision_nn, (84, 84), interpolation=cv2.INTER_AREA)

    # --- DIBUJO DE INTERFAZ ---
    if bbox_abs is not None:
        cv2.rectangle(frame, (bbox_abs[0], bbox_abs[1]), 
                             (bbox_abs[0] + bbox_abs[2], bbox_abs[1] + bbox_abs[3]), (0, 255, 0), 2)
                             
    # Calcular y mostrar FPS
    t_end = time.perf_counter()
    fps = 1 / (t_end - t_start)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(frame, "Vision NN (DeepLabV3)", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Ventanita de "Vision IA" 
    ai_vision_bgr = cv2.cvtColor(ai_vision_84, cv2.COLOR_GRAY2BGR)
    ai_display_size = 252 
    ai_vision_display = cv2.resize(ai_vision_bgr, (ai_display_size, ai_display_size), interpolation=cv2.INTER_NEAREST)
    frame[h_f-ai_display_size:h_f, w_f-ai_display_size:w_f] = ai_vision_display
    cv2.rectangle(frame, (w_f-ai_display_size, h_f-ai_display_size), (w_f, h_f), (0, 0, 255), 2)

    cv2.imshow("Calibrador Deep Learning", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
sct.close()