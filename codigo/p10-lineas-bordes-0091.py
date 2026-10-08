import cv2
import numpy as np

# EJEMPLO 2 — Detección de esquinas con Harris
# camello 0091
# Ian Gutierrez NC 0091

# Cargar imagen
imagen = cv2.imread("../imagenes/camello 0091.jpg")

# Verificar que la imagen exista
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(
    imagen,
    cv2.COLOR_BGR2GRAY
)

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
esquinas = cv2.dilate(
    esquinas,
    None
)

# Crear copia
resultado = imagen.copy()

# Umbral para identificar esquinas
umbral = 0.01 * esquinas.max()

# Marcar esquinas
resultado[esquinas > umbral] = [0, 0, 255]

# Mostrar resultados
cv2.imshow(
    "Imagen original - camello 0091",
    imagen
)

cv2.imshow(
    "Escala de grises - camello 0091",
    gris
)

cv2.imshow(
    "Esquinas detectadas - camello 0091",
    resultado
)

# Guardar escala de grises
cv2.imwrite(
    "../resultado2/camello_0091_grises.jpg",
    gris
)

# Guardar esquinas detectadas
cv2.imwrite(
    "../resultado2/camello_0091_esquinas.jpg",
    resultado
)

# Contar esquinas aproximadas
cantidad_esquinas = np.sum(
    esquinas > umbral
)

print("Deteccion de esquinas terminada.")

print("Escala de grises guardada en:")
print("../resultado2/camello_0091_grises.jpg")

print("Esquinas detectadas guardadas en:")
print("../resultado2/camello_0091_esquinas.jpg")

print(
    "Cantidad aproximada de puntos detectados:",
    cantidad_esquinas
)

# Esperar una tecla
cv2.waitKey(0)

# Cerrar ventanas
cv2.destroyAllWindows()

print("programa realizado por Ian Gutierrez NC = 0091")