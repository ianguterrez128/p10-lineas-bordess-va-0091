import cv2
import numpy as np

# EJEMPLO 2 — Detección de esquinas con Harris
# ian gutierrez NC 0091

# Cargar imagen
imagen = cv2.imread("../imagenes/camello 0091.jpg")

# Verificar que la imagen exista
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Convertir a tipo float32
gris_float = np.float32(gris)

# Detectar esquinas mediante Harris
esquinas = cv2.cornerHarris(
    gris_float,
    2,
    3,
    0.04
)

# Dilatar para hacer visibles las esquinas
esquinas = cv2.dilate(esquinas, None)

# Umbral
umbral = 0.01 * esquinas.max()

# Harris base
harris_base = imagen.copy()
harris_base[esquinas > umbral] = [0, 0, 255]

# Harris experimento
harris_experimento = imagen.copy()
harris_experimento[esquinas > umbral] = [0, 255, 0]

# Mostrar las 3 imágenes
cv2.imshow("Original camello 0091", imagen)

cv2.imshow("Harris base - Rojo camello 0091", harris_base)

cv2.imshow("Harris experimento - Verde camello 0091", harris_experimento)

# Guardar resultados
cv2.imwrite(
    "../resultados2/harris_base camello 0091.jpg",
    harris_base
)

cv2.imwrite(
    "../resultados2/harris_experimento camello 0091.jpg",
    harris_experimento
)

print("Deteccion de esquinas terminada.")
print("Harris base guardado en resultados2.")
print("Harris experimento guardado en resultados2.")

# Esperar una tecla
cv2.waitKey(0)

# Cerrar ventanas
cv2.destroyAllWindows()

print("programado por ian gutierrez NC = 0091")