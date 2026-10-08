import cv2
import numpy as np

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
esquinas = cv2.dilate(
    esquinas,
    None
)

# Crear copia
resultado1 = imagen.copy()
resultado2 = imagen.copy()

# Umbral para identificar esquinas
umbral = 0.01 * esquinas.max()
umbra2 = 0.05 * esquinas.max()

# Marcar esquinas
resultado1[esquinas > umbral] = [0, 0, 255]
resultado2[esquinas > umbra2] = [0, 0, 255]

# Mostrar resultados
cv2.imshow(
    "Imagen original camello 0091",
    imagen
)

cv2.imshow(
    "Esquinas detectadas camello 0091 (mas puntos)",
    resultado1
)
cv2.imshow(
    "Esquinas detectadas camello 0091 (menos puntos)",
    resultado2
)

# Guardar resultados
cv2.imwrite(
    "../resultados/camello 0091_esquinas1(mas puntos).jpg",
    resultado1
)
# Guardar resultados
cv2.imwrite(
    "../resultados/camello 0091_esquinas2(menos puntos).jpg",
    resultado2
)

# Contar esquinas aproximadas
cantidad_esquinas1 = np.sum(
    esquinas > umbral
    )

# Contar esquinas aproximadas
cantidad_esquinas2 = np.sum(
    esquinas > umbra2
    )

print("Deteccion de esquinas terminada.")
print("Cantidad aproximada de puntos detectados (umbral 0,01):",
      cantidad_esquinas1)
print("Deteccion de esquinas terminada.")
print("Cantidad aproximada de puntos detectados (umbral 0,05) 0091:",
      cantidad_esquinas2)

print("Resultados guardado en:")
print("../resultados/camello 0091_esquinas1(mas puntos).jpg")
print("../resultados/camello 0091_esquinas2(menos puntos).jpg")
# Esperar una tecla
cv2.waitKey(0)

# Cerrar ventanas
cv2.destroyAllWindows()

print("programa realizado por Ian Gutierrez NC = 0091")