# 📸 Técnicas de Procesamiento de Imágenes - Visión por Computadora I

## 📋 Índice de Contenidos

- [Clase 1: Fundamentos de OpenCV e Imágenes](#clase-1-fundamentos-de-opencv-e-imágenes)
- [Clase 2: Operadores de Píxel y Histogramas](#clase-2-operadores-de-píxel-y-histogramas)
- [Clase 3: Filtros y Transformadas](#clase-3-filtros-y-transformadas)
- [Clase 4: Detección de Patrones y Características](#clase-4-detección-de-patrones-y-características)
- [Clase 5: Descriptores de Características](#clase-5-descriptores-de-características)
- [Clase 6: Segmentación y Tracking](#clase-6-segmentación-y-tracking)
- [Clase 7: Procesamiento de Video y Flujo Óptico](#clase-7-procesamiento-de-video-y-flujo-óptico)

---

## 🔧 Clase 1: Fundamentos de OpenCV e Imágenes

### **Técnicas Principales:**

#### 1. **Carga y Visualización de Imágenes**
- **Para qué sirve**: Cargar imágenes desde archivos y mostrarlas en pantalla
- **Funciones clave**: `cv.imread()`, `cv.imshow()`, `plt.imshow()`
- **Ejemplo**:
```python
img = cv.imread('imagen.jpg', cv.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.show()
```

#### 2. **Manipulación de Canales de Color**
- **Para qué sirve**: Separar y trabajar con los diferentes canales RGB/BGR
- **Funciones clave**: `cv.split()`, `cv.merge()`
- **Ejemplo**:
```python
b, g, r = cv.split(img_color)  # Separar canales
img_merged = cv.merge([b, g, r])  # Reunir canales
```

#### 3. **Operaciones Básicas con Píxeles**
- **Para qué sirve**: Acceder y modificar valores individuales de píxeles
- **Ejemplo**:
```python
pixel_value = img[100, 150]  # Obtener valor de píxel
img[100, 150] = 255  # Modificar píxel
```

#### 4. **Conversión de Espacios de Color**
- **Para qué sirve**: Convertir entre diferentes representaciones de color (BGR, RGB, HSV, etc.)
- **Funciones clave**: `cv.cvtColor()`
- **Ejemplo**:
```python
img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
```

#### 5. **Segmentación por Color**
- **Para qué sirve**: Separar objetos de una imagen basándose en rangos de color
- **Funciones clave**: `cv.inRange()`, `cv.bitwise_and()`
- **Ejemplo**:
```python
# Detectar objetos verdes en HSV
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])
mask = cv.inRange(img_hsv, lower_green, upper_green)
result = cv.bitwise_and(img, img, mask=mask)
```

#### 6. **Operaciones Lógicas**
- **Para qué sirve**: Combinar imágenes usando operaciones booleanas
- **Funciones clave**: `cv.bitwise_and()`, `cv.bitwise_or()`, `cv.bitwise_not()`
- **Ejemplo**:
```python
result = cv.bitwise_and(img1, img2)  # AND entre dos imágenes
```

---

## 📊 Clase 2: Operadores de Píxel y Histogramas

### **Técnicas Principales:**

#### 1. **Control de Brillo**
- **Para qué sirve**: Aumentar o disminuir la luminosidad general de la imagen
- **Fórmula**: `nueva_imagen = imagen_original + (255 * porcentaje_brillo / 100)`
- **Ejemplo**:
```python
def change_brightness(img, bright_percent):
    img_new = img + (255 * bright_percent / 100)
    return np.clip(img_new, 0, 255).astype('uint8')

img_bright = change_brightness(img, 30)  # +30% brillo
```

#### 2. **Control de Contraste**
- **Para qué sirve**: Aumentar o disminuir la diferencia entre tonos claros y oscuros
- **Fórmula**: `nueva_imagen = (1 + contraste/100) * imagen_original`
- **Ejemplo**:
```python
def change_contrast(img, contrast_percent):
    img_new = (1 + contrast_percent / 100) * img
    return np.clip(img_new, 0, 255).astype('uint8')

img_contrast = change_contrast(img, 50)  # +50% contraste
```

#### 3. **Corrección Gamma**
- **Para qué sirve**: Ajustar la luminancia no linealmente para corregir el brillo
- **Cuándo usar**: Gamma < 1 para aclarar, Gamma > 1 para oscurecer
- **Ejemplo**:
```python
gamma = 0.7
img_gamma = np.power(img.astype('float32') / 255.0, 1.0/gamma)
img_gamma = (img_gamma * 255).astype('uint8')
```

#### 4. **Análisis de Histogramas**
- **Para qué sirve**: Analizar la distribución de intensidades en la imagen
- **Funciones clave**: `cv.calcHist()`, `np.histogram()`
- **Ejemplo**:
```python
hist = cv.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()
```

#### 5. **Histogramas 2D**
- **Para qué sirve**: Analizar la relación entre dos canales de color
- **Ejemplo**:
```python
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hist_2d = cv.calcHist([img_hsv], [0, 2], None, [180, 256], [0, 180, 0, 255])
plt.imshow(hist_2d, cmap='jet')
```

#### 6. **Ecualización de Histograma**
- **Para qué sirve**: Mejorar el contraste distribuyendo mejor las intensidades
- **Funciones clave**: `cv.equalizeHist()`
- **Ejemplo**:
```python
img_eq = cv.equalizeHist(img_gray)
```

#### 7. **Binarización (Thresholding)**
- **Para qué sirve**: Convertir imágenes a escala de grises en imágenes binarias
- **Tipos**: Fijo, Otsu, Adaptativo
- **Ejemplo**:
```python
# Umbral fijo
ret, thresh_fixed = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# Umbral Otsu (automático)
ret, thresh_otsu = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Umbral adaptativo
thresh_adaptive = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, 
                                       cv.THRESH_BINARY, 11, 2)
```

---

## 🔍 Clase 3: Filtros y Transformadas

### **Técnicas Principales:**

#### 1. **Filtro Gaussiano**
- **Para qué sirve**: Suavizar la imagen reduciendo el ruido
- **Funciones clave**: `cv.GaussianBlur()`, `cv.getGaussianKernel()`
- **Ejemplo**:
```python
img_blur = cv.GaussianBlur(img, (15, 15), sigmaX=5, sigmaY=5)
```

#### 2. **Filtros de Media y Mediana**
- **Para qué sirve**: 
  - **Media**: Suavizado general
  - **Mediana**: Eliminar ruido sal y pimienta específicamente
- **Ejemplo**:
```python
img_mean = cv.blur(img, (5, 5))  # Filtro de media
img_median = cv.medianBlur(img, 5)  # Filtro de mediana
```

#### 3. **Diferencia de Gaussianas (DoG)**
- **Para qué sirve**: Detectar bordes y características de diferentes escalas
- **Ejemplo**:
```python
blur1 = cv.GaussianBlur(img, (5, 5), 1.0)
blur2 = cv.GaussianBlur(img, (5, 5), 2.0)
dog = blur1 - blur2
```

#### 4. **Detección de Bordes con Canny**
- **Para qué sirve**: Detectar contornos y bordes con alta precisión
- **Funciones clave**: `cv.Canny()`
- **Ejemplo**:
```python
edges = cv.Canny(img, threshold1=100, threshold2=200, apertureSize=3)
```

#### 5. **Operadores de Gradiente (Sobel, Laplaciano)**
- **Para qué sirve**: Detectar cambios de intensidad en direcciones específicas
- **Ejemplo**:
```python
# Sobel en X e Y
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)

# Laplaciano
laplacian = cv.Laplacian(img, cv.CV_64F)
```

#### 6. **Transformada de Fourier**
- **Para qué sirve**: Análisis frecuencial de la imagen, filtrado en dominio de frecuencia
- **Ejemplo**:
```python
# FFT
f_transform = np.fft.fft2(img)
f_shift = np.fft.fftshift(f_transform)
magnitude = np.log(np.abs(f_shift) + 1)

# Filtros pasa-bajos y pasa-altos en frecuencia
def create_low_pass_filter(h, w, radius):
    center = (h//2, w//2)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    return (dist <= radius).astype(float)
```

#### 7. **Filtros Bilaterales**
- **Para qué sirve**: Suavizar manteniendo bordes nítidos
- **Funciones clave**: `cv.bilateralFilter()`
- **Ejemplo**:
```python
img_bilateral = cv.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
```

---

## 🎯 Clase 4: Detección de Patrones y Características

### **Técnicas Principales:**

#### 1. **Template Matching (Búsqueda de Patrones)**
- **Para qué sirve**: Encontrar un patrón específico dentro de una imagen más grande
- **Métodos**: TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
- **Ejemplo**:
```python
template = cv.imread('patron.png', 0)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Búsqueda de un solo objeto
res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

# Búsqueda de múltiples objetos
threshold = 0.8
locations = np.where(res >= threshold)
for pt in zip(*locations[::-1]):
    cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
```

#### 2. **Transformada de Hough para Líneas**
- **Para qué sirve**: Detectar líneas rectas en imágenes
- **Tipos**: Estándar y Probabilística
- **Ejemplo**:
```python
# Transformada de Hough estándar
edges = cv.Canny(img_gray, 50, 150)
lines = cv.HoughLines(edges, rho=1, theta=np.pi/180, threshold=200)

# Transformada de Hough probabilística
lines_p = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                         minLineLength=50, maxLineGap=10)
```

#### 3. **Transformada de Hough para Círculos**
- **Para qué sirve**: Detectar círculos en imágenes
- **Ejemplo**:
```python
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_blur = cv.medianBlur(img_gray, 5)

circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, dp=1, minDist=20,
                          param1=50, param2=30, minRadius=0, maxRadius=0)

for circle in circles[0, :]:
    cv.circle(img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
    cv.circle(img, (circle[0], circle[1]), 2, (0, 0, 255), 3)
```

#### 4. **Detección de Esquinas con Harris**
- **Para qué sirve**: Encontrar puntos de interés (esquinas) en la imagen
- **Ejemplo**:
```python
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray = np.float32(img_gray)

# Detector de Harris
dst = cv.cornerHarris(img_gray, blockSize=2, ksize=3, k=0.04)
dst = cv.dilate(dst, None)

# Marcar esquinas
img[dst > 0.01 * dst.max()] = [0, 0, 255]
```

#### 5. **Detector de Esquinas Shi-Tomasi**
- **Para qué sirve**: Alternativa mejorada al detector de Harris
- **Ejemplo**:
```python
corners = cv.goodFeaturesToTrack(img_gray, maxCorners=100, qualityLevel=0.01,
                                 minDistance=10, blockSize=3)

for corner in corners:
    x, y = corner.ravel()
    cv.circle(img, (x, y), 3, 255, -1)
```

#### 6. **Refinamiento Sub-píxel de Esquinas**
- **Para qué sirve**: Mejorar la precisión de la ubicación de las esquinas
- **Ejemplo**:
```python
criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 0.001)
corners_refined = cv.cornerSubPix(img_gray, np.float32(corners), 
                                  winSize=(5, 5), zeroZone=(-1, -1), 
                                  criteria=criteria)
```

#### 7. **Pirámides Gaussianas y Laplacianas**
- **Para qué sirve**: 
  - **Gaussianas**: Análisis multi-escala, reducción de resolución
  - **Laplacianas**: Compresión, fusión de imágenes
- **Ejemplo**:
```python
# Pirámide Gaussiana
def gaussian_pyramid(img, levels):
    pyramid = [img]
    for i in range(levels):
        img = cv.pyrDown(img)
        pyramid.append(img)
    return pyramid

# Pirámide Laplaciana
def laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid)-1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        expanded = cv.pyrUp(gaussian_pyramid[i+1], dstsize=size)
        laplacian = cv.subtract(gaussian_pyramid[i], expanded)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

# Fusión de imágenes usando pirámides
def blend_images(img1, img2, mask, levels):
    # Crear pirámides para ambas imágenes y la máscara
    gauss_pyr1 = gaussian_pyramid(img1, levels)
    gauss_pyr2 = gaussian_pyramid(img2, levels)
    mask_pyr = gaussian_pyramid(mask, levels)
    
    # Crear pirámides Laplacianas
    laplace_pyr1 = laplacian_pyramid(gauss_pyr1)
    laplace_pyr2 = laplacian_pyramid(gauss_pyr2)
    
    # Combinar usando la máscara
    blended_pyr = []
    for l1, l2, m in zip(laplace_pyr1, laplace_pyr2, mask_pyr):
        blended = l1 * m + l2 * (1.0 - m)
        blended_pyr.append(blended)
    
    return blended_pyr
```

---

## 🔍 Clase 5: Descriptores de Características

### **Técnicas Principales:**

#### 1. **SIFT (Scale-Invariant Feature Transform)**
- **Para qué sirve**: Detectar y describir características locales invariantes a escala y rotación
- **Ventajas**: Muy robusto, invariante a escala, rotación e iluminación
- **Ejemplo**:
```python
sift = cv.xfeatures2d.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img_gray, None)

# Dibujar keypoints
img_sift = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print(f'Descriptores encontrados: {len(keypoints)}')
print(f'Dimensión de descriptores: {descriptors.shape}')
```

#### 2. **SURF (Speeded-Up Robust Features)**
- **Para qué sirve**: Versión más rápida de SIFT con características similares
- **Ventajas**: Mayor velocidad que SIFT manteniendo robustez
- **Ejemplo**:
```python
surf = cv.xfeatures2d.SURF_create(hessianThreshold=400)
keypoints, descriptors = surf.detectAndCompute(img_gray, None)
img_surf = cv.drawKeypoints(img, keypoints, None)
```

#### 3. **ORB (Oriented FAST and Rotated BRIEF)**
- **Para qué sirve**: Alternativa rápida y libre de patentes a SIFT/SURF
- **Ventajas**: Muy rápido, libre de patentes, buenos resultados
- **Ejemplo**:
```python
orb = cv.ORB_create()
keypoints, descriptors = orb.detectAndCompute(img_gray, None)
img_orb = cv.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
```

#### 4. **FAST (Features from Accelerated Segment Test)**
- **Para qué sirve**: Detección rápida de esquinas
- **Ventajas**: Extremadamente rápido
- **Ejemplo**:
```python
fast = cv.FastFeatureDetector_create()
keypoints = fast.detect(img_gray, None)
img_fast = cv.drawKeypoints(img, keypoints, None, color=(255, 0, 0))
```

#### 5. **Matching de Características**
- **Para qué sirve**: Encontrar correspondencias entre características de dos imágenes
- **Métodos**: Brute Force, FLANN
- **Ejemplo**:
```python
# Brute Force Matcher
bf = cv.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Filtro de Lowe para buenos matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

# Dibujar matches
img_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
```

#### 6. **RANSAC para Homografía**
- **Para qué sirve**: Estimar transformaciones geométricas robustas eliminando outliers
- **Ejemplo**:
```python
# Extraer puntos de los matches
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Calcular homografía con RANSAC
M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

# Aplicar transformación
h, w = img1.shape[:2]
pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
dst = cv.perspectiveTransform(pts, M)
```

#### 7. **Creación de Panoramas**
- **Para qué sirve**: Unir múltiples imágenes para crear vistas panorámicas
- **Ejemplo**:
```python
def create_panorama(img1, img2):
    # Detectar características
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # Matching
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Filtrar matches
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    # Calcular homografía
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    
    # Crear panorama
    result = cv.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
    
    return result
```

---

## 🎨 Clase 6: Segmentación y Tracking

### **Técnicas Principales:**

#### 1. **Segmentación por K-Means**
- **Para qué sirve**: Agrupar píxeles similares en color para segmentar regiones
- **Ejemplo**:
```python
# Reformatear imagen para K-means
features = img.reshape((-1, 3))
features = np.float32(features)

# Criterios y configuración
criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 10, 1.0)
K = 5

# Aplicar K-means
compact, labels, centers = cv.kmeans(features, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Reconstruir imagen segmentada
centers = np.uint8(centers)
segmented_img = centers[labels.flatten()]
segmented_img = segmented_img.reshape(img.shape)
```

#### 2. **Algoritmo Watershed**
- **Para qué sirve**: Segmentación basada en morfología matemática, útil para separar objetos unidos
- **Ejemplo**:
```python
# Preprocesamiento
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Operaciones morfológicas
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# Área de fondo seguro
sure_bg = cv.dilate(opening, kernel, iterations=3)

# Área de primer plano seguro
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# Región desconocida
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marcadores para watershed
ret, markers = cv.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Aplicar watershed
markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]  # Marcar fronteras en rojo
```

#### 3. **Mean Shift Clustering**
- **Para qué sirve**: Segmentación automática sin especificar número de clusters
- **Ejemplo**:
```python
# Aplicar Mean Shift
shifted = cv.pyrMeanShiftFiltering(img, sp=21, sr=51)

# Convertir a array plano para análisis
flat_image = shifted.reshape((-1, 3))
flat_image = np.float32(flat_image)

# Aplicar K-means para obtener colores finales
criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 20, 1.0)
compactness, labels, centers = cv.kmeans(flat_image, 8, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
```

#### 4. **Mean Shift Tracking**
- **Para qué sirve**: Seguimiento de objetos basado en histogramas de color
- **Ejemplo**:
```python
# Configuración inicial
cap = cv.VideoCapture('video.mp4')
ret, frame = cap.read()

# Definir ROI inicial
x, y, w, h = 300, 200, 100, 50
track_window = (x, y, w, h)

# Calcular histograma de la ROI
roi = frame[y:y+h, x:x+w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# Criterios de terminación
term_crit = (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 10, 1)

while True:
    ret, frame = cap.read()
    if ret:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        
        # Aplicar Mean Shift
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        
        # Dibujar ventana de seguimiento
        x, y, w, h = track_window
        img2 = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        cv.imshow('Tracking', img2)
        
    if cv.waitKey(60) & 0xff == 27:
        break
```

#### 5. **CAMShift (Continuously Adaptive Mean Shift)**
- **Para qué sirve**: Extensión de Mean Shift que adapta el tamaño de la ventana de seguimiento
- **Ejemplo**:
```python
# Similar al Mean Shift pero usando CAMShift
ret, track_window = cv.CamShift(dst, track_window, term_crit)

# CAMShift devuelve un rectángulo rotado
pts = cv.boxPoints(ret)
pts = np.int0(pts)
img2 = cv.polylines(frame, [pts], True, 255, 2)
```

---

## 🎬 Clase 7: Procesamiento de Video y Flujo Óptico

### **Técnicas Principales:**

#### 1. **Flujo Óptico de Lucas-Kanade**
- **Para qué sirve**: Seguir características específicas entre frames consecutivos
- **Ejemplo**:
```python
# Parámetros para detección de características
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parámetros para Lucas-Kanade
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 10, 0.03))

# Capturar primer frame
cap = cv.VideoCapture('video.mp4')
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Crear máscara para dibujar estelas
mask = np.zeros_like(old_frame)
colors = np.random.randint(0, 255, (100, 3))

while True:
    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Calcular flujo óptico
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Seleccionar puntos buenos
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    # Dibujar estelas
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), colors[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, colors[i].tolist(), -1)
    
    img = cv.add(frame, mask)
    cv.imshow('Frame', img)
    
    # Actualizar frame anterior
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    
    if cv.waitKey(30) & 0xff == 27:
        break
```

#### 2. **Flujo Óptico Denso (Farneback)**
- **Para qué sirve**: Calcular el movimiento de todos los píxeles en la imagen
- **Ejemplo**:
```python
cap = cv.VideoCapture('video.mp4')
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    
    # Calcular flujo óptico denso
    flow = cv.calcOpticalFlowPyrLK(prvs, next_frame, None, None)
    
    # Convertir a coordenadas polares
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    
    # Convertir a BGR para visualización
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('Frame', bgr)
    
    if cv.waitKey(30) & 0xff == ord('q'):
        break
    
    prvs = next_frame
```

#### 3. **Sustracción de Fondo (Background Subtraction)**
- **Para qué sirve**: Detectar objetos en movimiento separándolos del fondo estático
- **Algoritmos**: MOG2, KNN

#### **Background Subtraction con MOG2**:
```python
# Crear el sustractor de fondo
back_sub = cv.createBackgroundSubtractorMOG2(detectShadows=True)

cap = cv.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if ret:
        # Aplicar sustractor de fondo
        fg_mask = back_sub.apply(frame)
        
        # Mostrar frame original y máscara
        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fg_mask)
        
        # Encontrar contornos en la máscara
        contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Dibujar contornos de objetos detectados
        for contour in contours:
            if cv.contourArea(contour) > 500:  # Filtrar objetos pequeños
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv.imshow('Objects', frame)
        
    if cv.waitKey(30) & 0xff == 27:
        break
```

#### **Background Subtraction con KNN**:
```python
# KNN es más robusto para escenas complejas
back_sub = cv.createBackgroundSubtractorKNN(detectShadows=True)

# El resto del código es similar al MOG2
```

#### 4. **Detección y Seguimiento de Múltiples Objetos**
- **Para qué sirve**: Rastrear varios objetos simultáneamente
- **Ejemplo**:
```python
# Usar background subtraction + tracking
class MultiObjectTracker:
    def __init__(self):
        self.back_sub = cv.createBackgroundSubtractorMOG2()
        self.trackers = []
    
    def detect_objects(self, frame):
        fg_mask = self.back_sub.apply(frame)
        contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            if cv.contourArea(contour) > 1000:
                x, y, w, h = cv.boundingRect(contour)
                objects.append((x, y, w, h))
        
        return objects
    
    def update_trackers(self, frame, detected_objects):
        # Actualizar trackers existentes y crear nuevos
        for obj in detected_objects:
            # Crear nuevo tracker para cada objeto detectado
            tracker = cv.TrackerCSRT_create()
            tracker.init(frame, obj)
            self.trackers.append(tracker)
```

#### 5. **Análisis de Movimiento y Actividad**
- **Para qué sirve**: Detectar patrones de movimiento, áreas de alta actividad
- **Ejemplo**:
```python
def motion_analysis(video_path):
    cap = cv.VideoCapture(video_path)
    
    # Acumulador para mapas de movimiento
    motion_history = None
    
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
            
        next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        
        # Calcular diferencia entre frames
        diff = cv.absdiff(prvs, next_frame)
        
        # Acumular movimiento
        if motion_history is None:
            motion_history = np.zeros_like(diff, dtype=np.float32)
        
        motion_history += diff.astype(np.float32)
        
        # Normalizar y mostrar mapa de calor
        motion_normalized = cv.normalize(motion_history, None, 0, 255, cv.NORM_MINMAX)
        motion_heatmap = cv.applyColorMap(motion_normalized.astype(np.uint8), cv.COLORMAP_JET)
        
        cv.imshow('Motion Heatmap', motion_heatmap)
        
        prvs = next_frame
        
        if cv.waitKey(30) & 0xff == 27:
            break
    
    return motion_history
```

---

## 📚 Resumen de Aplicaciones por Clase

### **🎯 Aplicaciones Principales por Técnica:**

| **Clase** | **Técnicas Principales** | **Aplicaciones Reales** |
|-----------|--------------------------|--------------------------|
| **1** | Manipulación básica, espacios de color | Preprocesamiento, segmentación por color, interfaces de usuario |
| **2** | Ajustes de imagen, histogramas, binarización | Mejora de calidad, análisis médico, OCR, visión nocturna |
| **3** | Filtros, detección de bordes, Fourier | Reducción de ruido, análisis de texturas, compresión de imágenes |
| **4** | Template matching, Hough, esquinas | Reconocimiento de objetos, detección de formas, calibración de cámaras |
| **5** | Descriptores robustos, matching | Panoramas, realidad aumentada, navegación de robots |
| **6** | Segmentación, tracking | Análisis médico, seguimiento de objetos, interfaces gestuales |
| **7** | Procesamiento de video, flujo óptico | Videovigilancia, análisis de tráfico, deportes, cinematografía |

---

## 🔧 Herramientas y Librerías Utilizadas

- **OpenCV (cv2)**: Librería principal para procesamiento de imágenes y visión por computadora
- **NumPy**: Operaciones matriciales y numéricas eficientes
- **Matplotlib**: Visualización de imágenes, gráficos e histogramas
- **Scikit-image**: Algoritmos adicionales de procesamiento de imágenes
- **Supervision**: Herramientas de visualización avanzadas

---

## 📖 Recursos Adicionales

Para profundizar en cada técnica, consulta:
- [Documentación oficial de OpenCV](https://docs.opencv.org/)
- [Tutoriales de OpenCV-Python](https://opencv-python-tutroials.readthedocs.io/)
- Papers académicos en la carpeta `Articulos/`
- Diapositivas del curso en `Diapositivas/`

---

*Este README fue creado como referencia completa de todas las técnicas de procesamiento de imágenes vistas en el curso de Visión por Computadora I.*
