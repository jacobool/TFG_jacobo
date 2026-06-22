import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import setup_paths  # noqa: F401, E402

"""
gd_death_detector.py  (v3 — independiente del color)
-----------------------------------------------------
El problema con detectar por color (verde, cian, blanco) es que
el fondo y la skin del jugador cambian en cada mapa. En un mapa
verde, la explosión también es verde/amarilla → imposible distinguir.

SOLUCIÓN: dos señales que son SIEMPRE consistentes, sin importar el mapa:

  SEÑAL 1 — Parada del movimiento (motion stop)
    El fondo de GD se mueve constantemente hacia la izquierda.
    Cuando el jugador muere, el fondo se congela ~1 segundo.
    Detectamos esto comparando la diferencia entre frames consecutivos
    en la zona del fondo (evitando la zona del jugador).

  SEÑAL 2 — Aparición del texto de intento ("PT X" / "1" / etc.)
    Al inicio de cada intento aparece un número/texto blanco con
    borde negro en la esquina superior izquierda. Detectamos esta
    transición de oscuro→blanco en esa zona.
    Esto detecta el RESPAWN, que es equivalente a saber que hubo muerte.

Estrategia combinada:
  - Muerte confirmada si: motion_stop AND (texto_aparece OR timeout_corto)
  - O si: texto_aparece directamente (más fiable como señal de respawn)
"""

import cv2
import numpy as np
import time
from collections import deque


# ── Tuning de motion stop ─────────────────────────────────────
# Diferencia media de píxeles entre frames (0-255).
# Durante el juego normal: ~8-20 (el fondo se mueve).
# Cuando mueres y el juego se congela: ~0-3.
MOTION_THRESHOLD     = 4.0    # Por debajo → "congelado"
MOTION_FRAMES_NEEDED = 3      # Nº de frames consecutivos congelados para confirmar
MOTION_SAMPLE_W      = 0.30   # Ancho de la zona de muestreo (fondo izquierdo)

# ── Tuning de detección de texto de intento ───────────────────
# El texto "PT X" aparece en la esquina superior izquierda.
# Buscamos una transición: zona oscura → zona con píxeles muy blancos.
TEXT_ZONE_X = (0.02, 0.25)    # fracción del ancho de pantalla
TEXT_ZONE_Y = (0.15, 0.35)    # fracción del alto de pantalla
TEXT_WHITE_THRESHOLD  = 200   # valor mínimo en escala de grises para "blanco"
TEXT_COVERAGE_MIN     = 0.04  # fracción mínima de píxeles blancos en la zona

# ── Cooldown entre muertes ────────────────────────────────────
DEATH_COOLDOWN_S = 1.2


class DeathDetector:
    """
    Detecta muerte en Geometry Dash de forma independiente al color,
    usando parada del movimiento del fondo y/o aparición del texto de intento.

    Uso:
        detector = DeathDetector()
        detector.reset()            # en env.reset()
        is_dead = detector.update(frame_bgr)   # en cada step
    """

    def __init__(self,
                 motion_threshold: float = MOTION_THRESHOLD,
                 motion_frames_needed: int = MOTION_FRAMES_NEEDED,
                 text_coverage_min: float = TEXT_COVERAGE_MIN,
                 cooldown_s: float = DEATH_COOLDOWN_S,
                 debug: bool = False):

        self.motion_threshold     = motion_threshold
        self.motion_frames_needed = motion_frames_needed
        self.text_coverage_min    = text_coverage_min
        self.cooldown_s           = cooldown_s
        self.debug                = debug

        self._prev_gray       = None
        self._frozen_count    = 0          # frames consecutivos sin movimiento
        self._last_death_time = 0.0
        self._prev_text_coverage = 0.0    # para detectar transición

    def reset(self):
        """Llamar en env.reset() al inicio de cada episodio."""
        self._prev_gray    = None
        self._frozen_count = 0
        self._prev_text_coverage = 0.0

    def update(self, frame_bgr: np.ndarray) -> bool:
        """
        Procesa un frame y devuelve True si detecta muerte/respawn.
        """
        now = time.time()
        if (now - self._last_death_time) < self.cooldown_s:
            return False

        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        motion_stopped = self._check_motion_stopped(gray, h, w)
        text_appeared  = self._check_attempt_text(gray, h, w)

        dead = False

        # Caso 1: texto de intento apareció → respawn confirmado (muerte anterior)
        if text_appeared:
            dead = True
            if self.debug:
                print("💀 [DD] RESPAWN detectado por texto de intento")

        # Caso 2: movimiento congelado N frames seguidos → muerte en curso
        elif motion_stopped:
            dead = True
            if self.debug:
                print("💀 [DD] MUERTE detectada por parada de movimiento")

        if dead:
            self._last_death_time = now
            self._frozen_count = 0
            self._prev_gray = None
            return True

        self._prev_gray = gray
        return False

    # ── Señal 1: Motion stop ──────────────────────────────────

    def _check_motion_stopped(self, gray, h, w) -> bool:
        """
        Compara el frame actual con el anterior en la zona del fondo
        (lejos del jugador). Si la diferencia media es muy baja durante
        varios frames seguidos → el juego está congelado → muerte.
        """
        if self._prev_gray is None:
            self._prev_gray = gray
            return False

        # Zona de muestreo: franja izquierda de la pantalla, lejos del jugador
        # (el jugador está al ~37% del ancho)
        x2 = int(w * MOTION_SAMPLE_W)
        y1 = int(h * 0.10)
        y2 = int(h * 0.85)

        curr_roi = gray[y1:y2, :x2].astype(np.float32)
        prev_roi = self._prev_gray[y1:y2, :x2].astype(np.float32)

        diff = np.abs(curr_roi - prev_roi).mean()

        if self.debug:
            print(f"[DD] motion_diff={diff:.2f}  frozen={self._frozen_count}")

        if diff < self.motion_threshold:
            self._frozen_count += 1
        else:
            self._frozen_count = 0

        return self._frozen_count >= self.motion_frames_needed

    # ── Señal 2: Texto de intento ─────────────────────────────

    def _check_attempt_text(self, gray, h, w) -> bool:
        """
        Detecta la transición oscuro→blanco en la zona donde aparece
        el número/texto de intento (esquina superior izquierda).
        Devuelve True en el frame en que el texto aparece súbitamente.
        """
        x1 = int(w * TEXT_ZONE_X[0])
        x2 = int(w * TEXT_ZONE_X[1])
        y1 = int(h * TEXT_ZONE_Y[0])
        y2 = int(h * TEXT_ZONE_Y[1])

        roi = gray[y1:y2, x1:x2]
        white_pixels = (roi >= TEXT_WHITE_THRESHOLD).sum()
        coverage = white_pixels / roi.size

        if self.debug:
            print(f"[DD] text_coverage={coverage:.4f}  prev={self._prev_text_coverage:.4f}")

        # Transición: antes había poco blanco, ahora hay mucho → texto apareció
        appeared = (coverage >= self.text_coverage_min and
                    self._prev_text_coverage < self.text_coverage_min * 0.5)

        self._prev_text_coverage = coverage
        return appeared


# ─────────────────────────────────────────────────────────────
# HERRAMIENTA DE CALIBRACIÓN
# Ejecuta:  python gd_death_detector.py
#
# Juega normalmente. Al morir verás en consola:
#   motion_diff      → debería caer a <4 cuando el juego se congela
#   text_coverage    → debería subir cuando aparece "PT X"
# Ajusta los umbrales según tus valores observados.
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import mss
    import pygetwindow as gw
    import ctypes
    from ctypes import wintypes

    def get_window_rect(hwnd):
        rect = wintypes.RECT()
        ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
        w2, h2 = rect.right - rect.left, rect.bottom - rect.top
        point = wintypes.POINT(0, 0)
        ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
        return {"top": point.y, "left": point.x, "width": w2, "height": h2}

    print("=" * 60)
    print("  CALIBRADOR v3 — Motion stop + texto de intento")
    print("=" * 60)
    print("Valores clave a observar:")
    print("  motion_diff  → normal: ~10-20  |  muerto: ~0-3")
    print("  text_coverage → normal: ~0     |  respawn: sube bruscamente")
    print("Pulsa Ctrl+C para salir.\n")

    wins = gw.getWindowsWithTitle("Geometry Dash")
    if not wins:
        print("❌ Ventana de Geometry Dash no encontrada.")
        exit(1)

    hwnd = wins[0]._hWnd
    sct = mss.mss()
    detector = DeathDetector(debug=True)

    try:
        while True:
            monitor = get_window_rect(hwnd)
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if detector.update(frame):
                print(">>> MUERTE / RESPAWN DETECTADO <<<\n")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nCalibración finalizada.")
        sct.close()