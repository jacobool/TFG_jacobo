import cv2
import numpy as np

IMAGE_PATH = "Captura de pantalla 2026-03-09 121433.png"  # Pon aquí tu captura

hsv_image = None

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_bgr = param[y, x]          # BGR
        pixel_hsv = hsv_image[y, x]      # HSV
        print(f"[CLICK] x={x}, y={y}  BGR={pixel_bgr}  HSV={pixel_hsv}")

def main():
    global hsv_image

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("No se pudo cargar la imagen. Revisa IMAGE_PATH.")
        return

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.namedWindow("GD_SAMPLE", cv2.WINDOW_NORMAL)
    cv2.imshow("GD_SAMPLE", img)
    cv2.setMouseCallback("GD_SAMPLE", mouse_callback, img)

    print("Haz clic sobre el borde VERDE del cubo varias veces.")
    print("Mira en la consola los valores HSV y apunta el rango de H, S y V.")

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
