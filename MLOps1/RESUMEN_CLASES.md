# Resumen de Clases - Operaciones de Aprendizaje Automático I

## 📋 Índice
- [Clase 1: Introducción y Buenas Prácticas](#clase-1-introducción-y-buenas-prácticas)
- [Clase 2: Desarrollo de Modelos y Docker](#clase-2-desarrollo-de-modelos-y-docker)
- [Clase 3: Infraestructura y MLFlow](#clase-3-infraestructura-y-mlflow)
- [Clase 4: Orquestación con Airflow](#clase-4-orquestación-con-airflow)
- [Clase 5: Despliegue de Modelos](#clase-5-despliegue-de-modelos)
- [Clase 6: APIs y Microservicios](#clase-6-apis-y-microservicios)
- [Clase 7: Modelos en Producción](#clase-7-modelos-en-producción)

---

## Clase 1: Introducción y Buenas Prácticas

### 🎯 Objetivos
Introducir los conceptos fundamentales de MLOps y establecer buenas prácticas de programación para proyectos de Machine Learning.

### 📚 Contenido Teórico
- **Introducción a la Materia**: Presentación del curso y objetivos
- **Ciclo de vida de un proyecto de Aprendizaje Automático**: Desde la investigación hasta la producción
- **Machine Learning Operations (MLOps)**: Definición, importancia y beneficios
- **Niveles de MLOps**: Diferentes grados de automatización y madurez
- **Entorno productivo**: Características y diferencias con el entorno de desarrollo
- **Buenas prácticas de programación**: Estándares de código, documentación y estructura

### 🛠️ Hands-on
**Refactorización de Notebook a Scripts Python**
- Migración de código desde Jupyter Notebook (`svm.ipynb`) a scripts modulares
- Aplicación de buenas prácticas:
  - Modularidad del código
  - Documentación adecuada
  - Nombramiento descriptivo de variables
  - Separación de responsabilidades
- Archivos resultantes:
  - `etl.py`: Procesamiento y limpieza de datos
  - `train_model.py`: Entrenamiento del modelo
  - `test_model.py`: Evaluación del modelo

### 🔑 Conceptos Clave
- Clean Code
- Separación de concerns
- Modularidad
- Documentación
- Testing

---

## Clase 2: Desarrollo de Modelos y Docker

### 🎯 Objetivos
Profundizar en las metodologías de desarrollo de modelos ML y introducir contenedores como herramienta de despliegue.

### 📚 Contenido Teórico
- **Desarrollo de modelos**: Metodologías y mejores prácticas
- **Seleccionar el tipo de modelo**: Criterios de selección según el problema
- **Las 4 fases del desarrollo de modelos**:
  1. Definición del problema
  2. Preparación de datos
  3. Entrenamiento y evaluación
  4. Despliegue
- **Depurando modelos**: Técnicas de debugging y optimización
- **Entrenamiento distribuido**: Estrategias para modelos grandes
- **Métodos de evaluación**: Métricas y validación cruzada
- **Contenedores y Docker**: Conceptos fundamentales de containerización

### 🛠️ Hands-on
**Experimentación con Docker**
- **1-simple_case/**: Creación de container básico con Python
- **2-simple_server/**: Servidor web containerizado con Flask
- **3-wordpress/**: Orquestación multi-container con docker-compose
- **4-mini-model-service/**: Servicio completo de ML con base de datos

### 🔑 Conceptos Clave
- Containerización
- Docker y docker-compose
- Microservicios
- Reproducibilidad
- Aislamiento de entornos

---

## Clase 3: Infraestructura y MLFlow

### 🎯 Objetivos
Comprender la infraestructura necesaria para ML en producción e introducir herramientas de gestión de experimentos.

### 📚 Contenido Teórico
- **Infraestructura**: Componentes necesarios para MLOps
- **Capa de almacenamiento**: Bases de datos, data lakes, object storage
- **Capa de cómputo**: CPU, GPU, clusters, cloud computing
- **Plataforma de ML**: Ecosistema integrado para ML
- **MLFlow**: Plataforma open-source para el ciclo de vida de ML

### 🛠️ Hands-on
**Gestión de Modelos con MLFlow**
- Configuración de MLFlow server con Docker
- Tracking de experimentos
- Registro y versionado de modelos
- Comparación de métricas entre experimentos
- Gestión de artefactos

### 🛠️ Jupyter Notebook
- **pandas_vs_numpy.ipynb**: Comparación de rendimiento entre librerías
- Datasets: `winequality-red.csv` y `winequality-red.parquet`

### 🔑 Conceptos Clave
- Experiment tracking
- Model registry
- Artifact storage
- Model versioning
- Reproducibilidad de experimentos

---

## Clase 4: Orquestación con Airflow

### 🎯 Objetivos
Implementar workflows automatizados para pipelines de ML usando Apache Airflow.

### 📚 Contenido Teórico
- **Administración de recursos**: Gestión eficiente de recursos computacionales
- **Orquestadores y sincronizadores**: Herramientas para automatización
- **Gestión del flujo de trabajo de ciencia de datos**: Pipelines end-to-end
- **Apache Airflow**: Plataforma de orquestación de workflows

### 🛠️ Hands-on
**Pipelines ETL con Airflow**
- Configuración de Airflow con Docker
- Creación de DAGs (Directed Acyclic Graphs)
- Implementación de pipelines ETL:
  - `etl_process_taskflow.py`: Usando TaskFlow API
  - `etl_process_tradicional.py`: Usando operadores tradicionales
- Gestión de conexiones y variables
- Monitoreo y logging

### 🔑 Conceptos Clave
- Workflow orchestration
- DAGs (Directed Acyclic Graphs)
- TaskFlow API
- Scheduling
- Dependency management

---

## Clase 5: Despliegue de Modelos

### 🎯 Objetivos
Explorar diferentes estrategias de despliegue de modelos, enfocándose en predicción por lotes.

### 📚 Contenido Teórico
- **Despliegue de modelos**: Estrategias y consideraciones
- **Estrategias de despliegue**: Blue-green, canary, rolling updates
- **Sirviendo modelos**: Diferentes modalidades de servicio
- **Propiedades del entorno de ejecución**: Latencia, throughput, disponibilidad
- **Predicción en lotes**: Procesamiento masivo de datos

### 🛠️ Hands-on
**Batch Processing con Metaflow y Redis**
- Configuración de entorno con docker-compose
- Implementación de pipeline batch con Metaflow
- Uso de Redis para caching y cola de tareas
- Almacenamiento con MinIO
- Base de datos PostgreSQL
- Procesamiento de dataset Iris

### 🔑 Conceptos Clave
- Batch processing
- Real-time vs batch inference
- Caching strategies
- Object storage
- Pipeline automation

---

## Clase 6: APIs y Microservicios

### 🎯 Objetivos
Implementar servicios web para modelos ML usando APIs REST y arquitectura de microservicios.

### 📚 Contenido Teórico
- **Despliegue de modelos**: Modalidades online
- **Desplegado on-line**: Servicios web en tiempo real
- **APIs y Microservicios**: Arquitectura distribuida
- **REST API**: Protocolo y mejores prácticas
- **Implementación de REST APIs en Python**: Usando FastAPI/Flask

### 🛠️ Hands-on
**Desarrollo de APIs para ML**
- Progresión de complejidad en APIs:
  - `main_1.py`: API básica
  - `main_2.py`: API con validación
  - `main_3.py`: API con modelo ML
  - `main_4.py`: API completa con logging
- Testing de APIs
- Documentación automática
- Validación de datos

### 🛠️ Notebooks
- **1-usando_una_API.ipynb**: Consumo de APIs externas
- **2-testing_API_creada.ipynb**: Testing de la API desarrollada

### 🔑 Conceptos Clave
- REST APIs
- FastAPI/Flask
- Request/Response validation
- API documentation
- Microservices architecture

---

## Clase 7: Modelos en Producción

### 🎯 Objetivos
Consolidar conocimientos sobre el despliegue de modelos en entornos productivos reales.

### 📚 Contenido Teórico
- **Sirviendo modelos en el mundo real**: Casos de uso y arquitecturas
- **Estrategias de implementación**: Patrones de despliegue avanzados
- **Ejemplo de servicios de modelos**: Casos prácticos y lecciones aprendidas

### 🔑 Conceptos Clave
- Production deployment
- Model serving patterns
- Scalability
- Monitoring and observability
- Performance optimization

---

## 🛠️ Tecnologías y Herramientas Utilizadas

### Lenguajes y Frameworks
- **Python** ≥3.10
- **FastAPI/Flask** para APIs
- **Scikit-learn** para ML
- **Pandas/NumPy** para manipulación de datos

### Infraestructura y Orquestación
- **Docker** y **docker-compose**
- **Apache Airflow** para workflows
- **MLFlow** para experiment tracking
- **Metaflow** para pipelines ML

### Storage y Bases de Datos
- **PostgreSQL** para datos relacionales
- **Redis** para caching
- **MinIO** para object storage

### Desarrollo y Testing
- **Jupyter Notebooks** para experimentación
- **pytest** para testing
- **Git** para control de versiones

---

## 📈 Progresión del Aprendizaje

1. **Fundamentos** (Clase 1-2): Buenas prácticas y containerización
2. **Infraestructura** (Clase 3-4): Plataformas ML y orquestación
3. **Despliegue** (Clase 5-6): Batch processing y APIs
4. **Producción** (Clase 7): Implementación en entornos reales

El curso progresa desde conceptos fundamentales hasta implementaciones complejas, proporcionando una base sólida para desarrollar y operar sistemas de Machine Learning en producción.
