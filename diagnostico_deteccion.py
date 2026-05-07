"""
diagnostico_deteccion.py
Herramienta de calibración para la detección de muerte.

Juega manualmente mientras este script:
  1. Muestra en tiempo real el valor de green_area superpuesto en pantalla
  2. Muestra la banda de detección y el bounding box del jugador
  3. Graba todos los valores a CSV para análisis posterior
  4. Marca los momentos de "muerte detectada" con el umbral actual

Al terminar (Ctrl+C), genera un gráfico con:
  - green_area a lo largo del tiempo
  - Línea del umbral actual (GREEN_AREA_MIN)
  - Zonas de muerte detectada marcadas en rojo

Así puedes ver EXACTAMENTE:
  - Qué green_area tiene el cubo cuando está vivo
  - Qué pasa durante la muerte (¿baja a 0? ¿baja gradualmente?)
  - Qué pasa durante el respawn (¿hay un dip transitorio?)
  - Si tu umbral actual está bien puesto o no

Uso: python diagnostico_deteccion.py
     (luego juega manualmente en Geometry Dash)
"""

import numpy as np
import cv2
import mss
import pygetwindow as gw
import ctypes
from ctypes import wintypes
import time
import csv
import sys

# ─── MISMA CONFIGURACIÓN QUE TU ENTORNO ──────────────────────────────────────
GAME_TITLE = "Geometry Dash"
PLAYER_X_REL = 0.345
PLAYER_BAND_W = 0.065
LOWER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
UPPER_GREEN = np.array([45, 255, 255], dtype=np.uint8)
GREEN_AREA_MIN = 600        # ← tu umbral actual
DEATH_FRAMES_NEEDED = 2     # ← tu valor actual
STEP_DURATION = 1 / 15      # más rápido que el entorno para capturar más datos

# ─── FUNCIONES ────────────────────────────────────────────────────────────────

def get_window_rect(hwnd):
    rect = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    point = wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
    return {"top": point.y, "left": point.x, "width": w, "height": h}


def detect_player_debug(frame_bgr):
    """Versión de detección que devuelve info extra para debug."""
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
    mask_clean = None
    contour_area = 0

    if band_bgr.size > 0:
        band_hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(band_hsv, LOWER_GREEN, UPPER_GREEN)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        green_area = cv2.countNonZero(mask_clean)
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = max(contours, key=cv2.contourArea) if contours else None

        if best_cnt is not None and cv2.contourArea(best_cnt) > 50:
            bx, by, bw, bh = cv2.boundingRect(best_cnt)
            player_center_y = y1 + by + bh // 2
            bbox_abs = (x1 + bx, y1 + by, bw, bh)
            contour_area = cv2.contourArea(best_cnt)

    band_coords = (x1, y1, x2, y2)
    return player_center_y, green_area, bbox_abs, band_coords, mask_clean, contour_area


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  DIAGNÓSTICO DE DETECCIÓN DE MUERTE")
    print("=" * 60)
    print("  1. Pon Geometry Dash en primer plano")
    print("  2. Juega manualmente (muere varias veces)")
    print("  3. Pulsa Ctrl+C para terminar y ver los resultados")
    print()
    print(f"  Umbral actual: GREEN_AREA_MIN = {GREEN_AREA_MIN}")
    print(f"  Frames para muerte: DEATH_FRAMES_NEEDED = {DEATH_FRAMES_NEEDED}")
    print("=" * 60)

    windows = gw.getWindowsWithTitle(GAME_TITLE)
    if not windows:
        print("❌ No se encontró la ventana de Geometry Dash")
        sys.exit(1)

    win = windows[0]
    hwnd = win._hWnd
    sct = mss.mss()

    # Datos para el CSV y gráficos
    log = []
    no_green_count = 0
    frame_idx = 0
    prev_gray = None
    start_time = time.perf_counter()

    print("\n🔍 Capturando datos... Juega y muere varias veces.\n")

    try:
        while True:
            step_start = time.perf_counter()

            # Capturar frame
            monitor = get_window_rect(hwnd)
            img = np.array(sct.grab(monitor))
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Detección
            player_y, green_area, bbox, band, mask, cnt_area = detect_player_debug(frame_bgr)

            # Frame difference
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frame_diff = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(frame_gray, prev_gray)
                frame_diff = float(np.mean(diff))
            prev_gray = frame_gray.copy()

            # Estado de muerte
            player_visible = green_area >= GREEN_AREA_MIN
            if not player_visible:
                no_green_count += 1
            else:
                no_green_count = 0

            death_triggered = no_green_count >= DEATH_FRAMES_NEEDED
            elapsed = time.perf_counter() - start_time

            # Log
            log.append({
                'frame': frame_idx,
                'time': round(elapsed, 4),
                'green_area': green_area,
                'contour_area': round(cnt_area, 1),
                'player_y': round(player_y, 1),
                'frame_diff': round(frame_diff, 2),
                'no_green_streak': no_green_count,
                'death_triggered': death_triggered,
                'player_visible': player_visible,
            })

            # Visualización en ventana de debug
            debug_frame = frame_bgr.copy()
            x1, y1, x2, y2 = band

            # Dibujar banda de detección
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

            # Dibujar bbox del jugador si existe
            if bbox is not None:
                bx, by, bw, bh = bbox
                cv2.rectangle(debug_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

            # Color del indicador según estado
            if death_triggered:
                color = (0, 0, 255)      # rojo = muerte detectada
                status = "MUERTE"
            elif not player_visible:
                color = (0, 165, 255)    # naranja = sin verde (contando)
                status = f"SIN VERDE x{no_green_count}"
            else:
                color = (0, 255, 0)      # verde = jugador visible
                status = "VIVO"

            # Texto informativo
            h_frame = frame_bgr.shape[0]
            cv2.putText(debug_frame, f"green_area: {green_area}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(debug_frame, f"umbral: {GREEN_AREA_MIN}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(debug_frame, f"contorno: {cnt_area:.0f}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(debug_frame, f"frame_diff: {frame_diff:.1f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(debug_frame, status, (10, h_frame - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Barra visual de green_area (proporción respecto al umbral)
            bar_w = min(int(green_area / max(GREEN_AREA_MIN, 1) * 200), 400)
            cv2.rectangle(debug_frame, (10, h_frame - 50), (10 + bar_w, h_frame - 35), color, -1)
            cv2.rectangle(debug_frame, (10, h_frame - 50), (210, h_frame - 35), (100, 100, 100), 1)
            # Línea del umbral en la barra
            cv2.line(debug_frame, (210, h_frame - 52), (210, h_frame - 33), (0, 0, 255), 2)

            # Mostrar ventana debug
            scale = 0.6
            small = cv2.resize(debug_frame, None, fx=scale, fy=scale)
            cv2.imshow("Diagnostico Deteccion", small)

            # Mostrar máscara verde si existe
            if mask is not None:
                cv2.imshow("Mascara Verde", mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1
            elapsed_step = time.perf_counter() - step_start
            if STEP_DURATION - elapsed_step > 0:
                time.sleep(STEP_DURATION - elapsed_step)

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    sct.close()

    if not log:
        print("No se capturaron datos.")
        sys.exit(0)

    # ── Guardar CSV ──────────────────────────────────────────────────────────
    csv_path = "diagnostico_deteccion.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)
    print(f"\n💾 Datos guardados: {csv_path} ({len(log)} frames)")

    # ── Generar gráfico ──────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        times = [r['time'] for r in log]
        greens = [r['green_area'] for r in log]
        diffs = [r['frame_diff'] for r in log]
        deaths = [r['time'] for r in log if r['death_triggered']]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Gráfico 1: green_area
        ax1.plot(times, greens, color='green', linewidth=0.8, alpha=0.8)
        ax1.axhline(y=GREEN_AREA_MIN, color='red', linestyle='--', linewidth=1.5,
                     label=f'Umbral actual ({GREEN_AREA_MIN})')
        ax1.fill_between(times, 0, greens, alpha=0.15, color='green')

        # Marcar muertes detectadas
        for dt in deaths:
            ax1.axvline(x=dt, color='red', alpha=0.3, linewidth=1)

        ax1.set_ylabel('green_area (píxeles)')
        ax1.set_title('Diagnóstico de detección — green_area vs tiempo')
        ax1.legend()
        ax1.set_ylim(bottom=0)

        # Estadísticas en el gráfico
        alive_greens = [r['green_area'] for r in log if r['green_area'] >= GREEN_AREA_MIN]
        dead_greens = [r['green_area'] for r in log if r['green_area'] < GREEN_AREA_MIN]
        if alive_greens:
            ax1.text(0.02, 0.95,
                     f"Vivo: media={np.mean(alive_greens):.0f}, "
                     f"min={np.min(alive_greens)}, max={np.max(alive_greens)}",
                     transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        if dead_greens:
            ax1.text(0.02, 0.82,
                     f"Muerto: media={np.mean(dead_greens):.0f}, "
                     f"min={np.min(dead_greens)}, max={np.max(dead_greens)}",
                     transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        # Gráfico 2: frame difference
        ax2.plot(times, diffs, color='purple', linewidth=0.8, alpha=0.8)
        ax2.fill_between(times, 0, diffs, alpha=0.1, color='purple')
        for dt in deaths:
            ax2.axvline(x=dt, color='red', alpha=0.3, linewidth=1)
        ax2.set_ylabel('Frame difference (media)')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_title('Frame difference — picos = cambios bruscos (muerte/respawn)')

        plt.tight_layout()
        plot_path = "diagnostico_deteccion.png"
        plt.savefig(plot_path, dpi=150)
        print(f"📊 Gráfico guardado: {plot_path}")
        plt.show()

    except ImportError:
        print("⚠ matplotlib no disponible, solo se generó el CSV.")

    # ── Recomendaciones automáticas ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RECOMENDACIONES")
    print("=" * 60)

    alive_values = [r['green_area'] for r in log if r['green_area'] > 100]
    dead_values = [r['green_area'] for r in log if r['green_area'] <= 100]

    if alive_values:
        p5 = np.percentile(alive_values, 5)
        p1 = np.percentile(alive_values, 1)
        mean_alive = np.mean(alive_values)
        print(f"  green_area cuando el jugador está vivo:")
        print(f"    Media: {mean_alive:.0f}")
        print(f"    Percentil 5: {p5:.0f}")
        print(f"    Percentil 1: {p1:.0f}")
        print(f"    Mínimo: {np.min(alive_values)}")
        print()

        suggested = int(p1 * 0.7)
        print(f"  Umbral sugerido: GREEN_AREA_MIN = {suggested}")
        print(f"    (70% del percentil 1 de valores vivos)")
        if suggested != GREEN_AREA_MIN:
            print(f"    Tu valor actual ({GREEN_AREA_MIN}) "
                  f"{'es demasiado alto' if GREEN_AREA_MIN > suggested else 'es demasiado bajo'}")

    if dead_values:
        print(f"\n  green_area cuando el jugador NO está (muerte/respawn):")
        print(f"    Media: {np.mean(dead_values):.0f}")
        print(f"    Máximo: {np.max(dead_values)}")

    # Análisis de frame_diff en muertes
    death_diffs = [r['frame_diff'] for r in log if r['death_triggered']]
    if death_diffs:
        print(f"\n  frame_diff en momentos de muerte:")
        print(f"    Media: {np.mean(death_diffs):.1f}")
        print(f"    Usar como segunda señal si > {np.mean(death_diffs) * 0.5:.0f}")

    print("=" * 60)
