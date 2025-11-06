import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Cargo imagen ---
img_path = os.path.join(os.getcwd(), 'placa.png') #devuelve ruta del directorio actual
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 

if img is None:
    raise ValueError("La imagen no se pudo cargar. Verifica el path o nombre del archivo.")

#plt.figure(), plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title('Imagen Original'), plt.show(block=False)

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)


# Suavizado para reducir ruido antes de calcular gradientes
img_suavizada = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)

# Cálculo del gradiente
ddepth = cv2.CV_64F
grad_x = cv2.Sobel(img_suavizada, ddepth, 1, 0, ksize=3)
grad_y = cv2.Sobel(img_suavizada, ddepth, 0, 1, ksize=3)

# Magnitud del gradiente
grad = np.sqrt(grad_x*2 + grad_y*2)

# Casting para visualización
grad_n = cv2.convertScaleAbs(grad)

#binarizado con umbral
_, grad_bin = cv2.threshold(grad_n, 50, 255, cv2.THRESH_BINARY)

# Mostrar resultado
#imshow(grad_n, color_img=False, title="Gradiente con suavizado gaussiano")
#imshow(grad_bin, color_img=False, title="Gradiente binarizadoo")
# ---- Clausura (Closing) -----------------------
B = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 12))
img_clau = cv2.morphologyEx(grad_bin, cv2.MORPH_CLOSE, B)

#plt.figure()
#ax1 = plt.subplot(121); imshow(grad_n, new_fig=False, title="Original")
#plt.subplot(122, sharex=ax1, sharey=ax1); imshow(img_clau, new_fig=False, title="Clausura")
#plt.show(block=False)

# --- Erosion (Erode) ---------------------------
L = 8  
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (L, L) )
img_er = cv2.erode(img_clau, kernel, iterations=1)

#plt.figure()
#ax1 = plt.subplot(121); imshow(F, new_fig=False, title="Original")
#plt.subplot(122, sharex=ax1, sharey=ax1); imshow(img_er, new_fig=False, title="Erosion")
#plt.show(block=False)


# ---- Componentes conectados -----------------------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_er,
connectivity, cv2.CV_32S)

# ---- Bounding box ------- 
MIN_AREA = 2000
img_bboxes = img.copy() 

for i in range(1, num_labels):  # ignoramos fondo
    if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
        continue
    x, y, w, h = stats[i, :4]
    cx, cy = centroids[i]
    cv2.rectangle(img_bboxes, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    cv2.circle(img_bboxes, (int(cx), int(cy)), 6, color=(0, 0, 255), thickness=-1)

#imshow(img=img_bboxes, color_img=True, title="Bounding Boxes sobre imagen original")

# ----- DETECCION COMPONENTES PLACA -----
# Copia color de la imagen original para distintas visualizaciones
img_resistencias = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
img_clasificada = img_resistencias.copy()
img_separacion = img_resistencias.copy()  # Para mostrar resistencias y capacitores juntos

# Parámetros generales
H, W = img.shape[:2]
margin_x = int(W * 0.025)
margin_y = int(H * 0.025)
margin_x_r = int(W * 0.035)
margin_y_r = int(H * 0.035)

# ----- DETECCIÓN DE RESISTENCIAS -----
AREA_MIN = 2500
AREA_MAX = 15000
ASPECT_RATIO_TH = 2.5
ASPECT_RATIO_CHIP = 1.5
cant_resistencias = 0
AREA_CHIP = 36000

for i in range(1, num_labels):
    x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
    w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]

    # Filtros de bordes 
    if (x <= margin_x_r or y <= margin_y_r or x + w >= W - margin_x_r or y + h >= H - margin_y_r):
        continue

    # Proporción ancho y altura
    aspect_ratio = max(w, h) / min(w, h)

    # --- DETECCIÓN DE CHIP (área > 38000) ---
    if area > AREA_CHIP and aspect_ratio > ASPECT_RATIO_CHIP:
        color = (0, 255, 255)  # Amarillo
        cv2.rectangle(img_resistencias, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img_resistencias, "CHIP", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(img_separacion, (x, y), (x + w, y + h), color, 2)
        continue  # Salta el resto de los filtros de resistencias
    
    # Filtro de tamaño
    if area < AREA_MIN or area > AREA_MAX:
        continue
    
    # Filtro de intensidad
    roi = img[y:y+h, x:x+w]
    obj_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
    obj_pixels = roi[obj_mask == 1]
    mean_intensity = obj_pixels.mean() if len(obj_pixels) > 0 else 0
    if mean_intensity < 100:
        continue
    
    # Filtro de porcentaje de pixels
    bbox_area = w * h
    occupancy_ratio = area / bbox_area
    if occupancy_ratio < 0.3:
        continue
    
    # Filtro de proporción ancho y altura
    if aspect_ratio <= ASPECT_RATIO_TH:
        continue

    # Es una resistencia
    color = (0, 255, 0)  # Verde
    cv2.rectangle(img_resistencias, (x, y), (x + w, y + h), color, 2)
    cv2.circle(img_resistencias, (int(centroids[i][0]), int(centroids[i][1])), 4, color, -1)

    # Dibujar también en imagen de separación
    cv2.rectangle(img_separacion, (x, y), (x + w, y + h), color, 2)
    cant_resistencias += 1

# ----- DETECCIÓN GENERAL DE CAPACITORES  -----
RHO_TH = 0.35
AREA_TH = 20000
SQUARE_TOL = 0.2

for i in range(1, num_labels):
    x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
    w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

    if (x <= margin_x or y <= margin_y or x + w >= W - margin_x or y + h >= H - margin_y):
        continue

    area_bbox = w * h
    if area_bbox < AREA_TH:
        continue

    obj = (labels == i).astype(np.uint8)
    contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        continue

    area_contour = cv2.contourArea(contours[0])
    perimeter = cv2.arcLength(contours[0], True)
    if perimeter == 0:
        continue

    rho = 4 * np.pi * area_contour / (perimeter ** 2)
    ratio = w / h
    flag_circular = rho > RHO_TH
    flag_square_like = (1 - SQUARE_TOL) <= ratio <= (1 + SQUARE_TOL)

    if flag_circular or flag_square_like:
        # Dibujar en imagen de separación
        color = (0, 0, 255)  # Rojo
        cv2.rectangle(img_separacion, (x, y), (x + w, y + h), color, 2)

# ----- CLASIFICACIÓN DE CAPACITORES (por tamaño) -----
cant_pequeños = 0
cant_medianos = 0
cant_grandes = 0
TAM_PEQ = 80000
TAM_MED = 170000

for i in range(1, num_labels):
    x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
    w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

    if (x <= margin_x or y <= margin_y or x + w >= W - margin_x or y + h >= H - margin_y):
        continue

    area_bbox = w * h
    if area_bbox < AREA_TH:
        continue

    obj = (labels == i).astype(np.uint8)
    contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        continue

    area_contour = cv2.contourArea(contours[0])
    perimeter = cv2.arcLength(contours[0], True)
    if perimeter == 0:
        continue

    rho = 4 * np.pi * area_contour / (perimeter ** 2)
    ratio = w / h
    flag_circular = rho > RHO_TH
    flag_square_like = (1 - SQUARE_TOL) <= ratio <= (1 + SQUARE_TOL)

    if flag_circular or flag_square_like:
        if area_bbox < TAM_PEQ:
            cant_pequeños += 1
            color = (0, 255, 0)
            label = "P"
        elif area_bbox < TAM_MED:
            cant_medianos += 1
            color = (255, 0, 0)
            label = "M"
        else:
            cant_grandes += 1
            color = (255, 0, 255)
            label = "G"

        cv2.rectangle(img_clasificada, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img_clasificada, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


# ----- VISUALIZACIONES -----
plt.figure()
plt.imshow(cv2.cvtColor(img_separacion, cv2.COLOR_BGR2RGB))
plt.title("Separación de resistencias (verde), capacitores (rojo) y chip (amarillo)")
plt.axis("off")

plt.figure()
plt.imshow(cv2.cvtColor(img_clasificada, cv2.COLOR_BGR2RGB))
plt.title("Clasificación de capacitores:\nVerde=pequeño, Azul=mediano, Magenta=grande")
plt.axis("off")

# ----- RESULTADOS -----
print(f"Cantidad de resistencias detectadas: {cant_resistencias}")
print("Capacitores detectados:")
print(f"- Grandes  : {cant_grandes}")
print(f"- Medianos : {cant_medianos}")
print(f"- Pequeños : {cant_pequeños}")

plt.show()