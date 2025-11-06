## Detección de Componentes Electrónicos

##  Descripción
Este proyecto forma parte del **Trabajo Práctico N°2 de la materia Procesamiento Digital de Imágenes (TUIA - UNR)**.  
El objetivo del ejercicio es **detectar y clasificar automáticamente los componentes electrónicos** presentes en una placa PCB, a partir de una imagen en color (`placa.png`).

A través del procesamiento digital de imágenes, se implementan técnicas de filtrado, umbralado, morfología y análisis de componentes conectadas para **identificar resistencias, capacitores y circuitos integrados (chips)**.

---

##  Flujo del algoritmo

1. **Lectura y preprocesamiento**
   - Carga de la imagen original en color.
   - Conversión a escala de grises.
   - Aplicación de un filtro gaussiano para suavizar ruido.

2. **Detección de bordes y binarización**
   - Cálculo de gradiente o uso del método Canny.
   - Binarización por umbral adaptativo.

3. **Operaciones morfológicas**
   - Clausura para unir bordes interrumpidos.
   - Erosión para separar componentes próximos.

4. **Etiquetado de componentes conectados**
   - Identificación de regiones con `cv2.connectedComponents`.
   - Extracción de área, perímetro y bounding box.

5. **Clasificación por propiedades geométricas**
   - **Resistencias**: formas alargadas y áreas pequeñas.
   - **Capacitores**: regiones circulares o elípticas.
   - **Chip**: región rectangular grande (mayor área).

6. **Visualización**
   - Dibujo de bounding boxes codificados por color:
     - 🟩 Resistencias  
     - 🟥 Capacitores  
     - 🟨 Chip  

---
## Estructura del repositorio

Deteccion-Componentes-Placa/
│
├── deteccion_componentes.py        # Código principal del ejercicio
│
├── input/
│   └── placa.png                       # Imagen original de la placa PCB
│
├── output/
│   ├── componentes.png                 # Componentes detectados
│   └── clasificacion.png               # Resultado final con bounding boxes
│
├── requirements.txt                    # Librerías necesarias para ejecutar el proyecto
└── README.md                           # Documentación del ejercicio

## Librerías utilizadas
- `OpenCV (cv2)` – procesamiento de imágenes, filtros y morfología  
- `NumPy` – manipulación de matrices y operaciones lógicas  
- `Matplotlib` – visualización de resultados  
- *(Opcional)* `scipy.ndimage` – operaciones morfológicas adicionales  

---

## 📂 Estructura del repositorio
