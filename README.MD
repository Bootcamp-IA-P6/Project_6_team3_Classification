#  Data Science and Animal Prevention
  

### 📋 Descripción
**Refugio** 🐾 Este sistema es una plataforma inteligente de gestión integral diseñada para refugios de animales modernos. No se limita a ser una base de datos estática; es un ecosistema digital que utiliza Machine Learning para analizar el historial de cada rescatado.

A través de una arquitectura robusta con Supabase y Docker, el sistema centraliza la información médica, comportamental y logística. El núcleo del proyecto es un modelo predictivo desarrollado en Python que evalúa las probabilidades de adopción en tiempo real, permitiendo a los administradores tomar decisiones basadas en datos y no solo en intuición.

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-CD792C?style=for-the-badge&logo=polars&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=matplotlib&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)

[Características](#-características-principales) • [Instalación](#-instalación-local) • [Análisis de Datos](#-análisis-del-modelo-de-datos) • [API](#-endpoints-principales) • [Despliegue](#-dockerización)

### 🎯 Objetivo del Proyecto
Proporcionar una herramienta administrativa eficiente que permita gestionar el ciclo de vida de los animales (ingreso, seguimiento médico, socialización y adopción) y la programación de jornadas de rescate o visitas, garantizando la integridad de los datos mediante una arquitectura moderna, segura y escalable.

---


## 📊 Análisis de la Variable Objetivo: OutcomeClass



** ** 
La gráfica presenta el destino final de los animales en el refugio, dividida en cuatro categorías: Adoption (Adopción), Transfer (Transferencia), Return to Owner (Regreso al dueño) y At Risk (En riesgo).

1. Hallazgos Principales (Insights)
Dominio de la Adopción (49%): Casi la mitad de los casos resultan en adopciones exitosas (54,387 animales). Esto es positivo, ya que el modelo tiene suficientes ejemplos para aprender qué hace que un animal sea "adoptable".

Transferencias Relevantes (28.2%): Un volumen alto de animales es trasladado a otras organizaciones. Esto sugiere que el refugio colabora activamente con otras redes.

Desequilibrio Crítico en "At Risk" (3.7%): Esta es la clase minoritaria (solo 4,120 casos). Técnicamente, esto significa que el modelo podría tener dificultades para predecir cuándo un animal está en riesgo real, ya que tiene muy pocos ejemplos en comparación con las adopciones.

Regreso al Dueño (19.1%): Aproximadamente 1 de cada 5 animales se pierde y logra reencontrarse con su familia original.

  <img src="https://github.com/Bootcamp-IA-P6/Project_6_team3_Classification/blob/feat/ensemble_1/reports/figures/01_class_distribution.png?raw=true" width="850">  
---

### 💡 Beneficios Técnicos del Modelo Predictivo

**1. Optimización del "Flujo de Vida" Animal**

 ``Intervención Proactiva:`` Al identificar animales con baja probabilidad de adopción desde el día 1, el refugio puede asignarles recursos especiales (entrenamiento, mejores fotos, publicidad pagada) antes de que se vuelvan "residentes de larga estancia".

``Gestión de Expectativas:`` Permite al personal saber de antemano qué animales tienen más probabilidades de ser devueltos o transferidos, preparando mejor a los futuros dueños.
**Toma de Decisiones Basada en Evidencia**

``Identificación de Patrones Ocultos:`` Quizás el refugio no había notado que ciertos colores o razas en combinación con una edad específica tienen dificultades para ser adoptados. El modelo revela estas correlaciones automáticamente.

``Validación de Resultados:`` Mediante la Matriz de Confusión, puedes demostrar con números qué tan confiables son las predicciones, lo cual es vital para solicitar fondos o donaciones.

**Eficiencia en el Uso de Recursos Limitados** 
``Ahorro en Mantenimiento:`` Cada día que un animal pasa en el refugio tiene un costo (comida, limpieza, salud). Reducir el promedio de días de estancia mediante predicciones acertadas libera presupuesto para rescatar a más animales.

``Planificación de Espacio:`` Si el modelo predice una ola de "Retornos al dueño" o "Transferencias", el administrador puede prever espacios vacíos para nuevas emergencias.

**Mejora en la Tasa de Éxito de Adopciones** 

``Reducción de Reingresos:`` Al entender qué factores llevan a un "At Risk" (En Riesgo), el refugio puede ser más estricto en los filtros de adopción para esos casos específicos, asegurando que el animal no sufra el trauma de ser devuelto.

``Segmentación de Adoptantes:`` Puedes usar los perfiles que el modelo marca como exitosos para buscar familias que encajen perfectamente con las necesidades de cada animal.

---





### ✨ Características Principales


---

### 🚀 Tecnologías

| Categoría | Tecnologías |
| :--- | :--- |
| **Runtime** | Python 3.10+ |
| **Base de Datos** | Supabase (PostgreSQL) |
| **Contenerización** | Docker, Docker Compose |
| **Servidor ASGI** | Streamlit |


---

### 📦 Instalación Local

**1️⃣ Clonar el repositorio**
bash
git clone [Refugio ](https://github.com/Bootcamp-IA-P6/Project_6_team3_Classification.git)


## Configurar variables de entorno Crea un archivo .env en la raíz del proyecto con tus credenciales de Supabase:

## Fragmento de código

* SUPABASE_URL=tu_url_de_supabase
* SUPABASE_KEY=tu_anon_key
* DATABASE_URL=postgresql://postgres:password@db.supabase.co:5432/postgres

3️⃣ Instalar dependencias
Bash
uv add -r requirements.txt

4️⃣ Ejecutar la aplicación

Bash
streamlit run app.py
💡 La documentación automática estará disponible en: http://localhost:8000/docs

## 🐳 Dockerización
El proyecto está completamente preparado para entornos de contenedores, lo que garantiza que funcione de manera idéntica en cualquier máquina.

Dockerfile: Configura la imagen base de Python, instala dependencias y optimiza el entorno de ejecución.

Docker Compose: Orquesta el backend, mapea los puertos (8000:8000) y carga automáticamente las variables de entorno.

## Comandos rápidos:

Bash
### Construir la imagen
docker-compose build

### Levantar el sistema
docker-compose up -d


📁 Estructura del Proyecto/ Refugio animals 

```
# 📂 Estructura del Proyecto: Refugio de Animales

Este proyecto sigue una arquitectura modular para garantizar la reproducibilidad y el orden en el entrenamiento del modelo.

```text
PROJECT_6_TEAM3_CLASSIFICATION/
├── 📁 .github/              # Automatizaciones y flujos de trabajo de GitHub Actions
├── 📁 .venv/                # Entorno virtual de Python (librerías aisladas)
├── 📁 data/                 # Datasets del refugio (Raw, Processed, Intermediary)
├── 📁 models/               # Modelos entrenados (.pkl, .h5) listos para producción
├── 📁 notebooks/            # Cuadernos de Jupyter (.ipynb) para análisis y pruebas
├── 📁 reports/              # Gráficos, matrices de confusión y reportes finales
├── 📁 src/                  # Código fuente modular (limpieza de datos, entrenamiento)
├── 📁 tests/                # Pruebas unitarias para asegurar que el código no falle
├── 📄 .gitignore            # Archivos que Git debe ignorar (como el .venv)
├── 📄 .python-version       # Define la versión específica de Python utilizada
├── 📄 main.py               # Punto de entrada principal para ejecutar el modelo
├── 📄 pyproject.toml        # Configuración de dependencias y herramientas del proyecto
├── 📄 README.md             # Documentación general del proyecto
└── 📄 uv.lock               # Archivo de bloqueo para la gestión exacta de dependencias (UV)
```
## 📝 Scripts Disponibles
npm run dev (o el comando equivalente en Python):

uvicorn app:app --reload: Inicia el servidor de desarrollo.

pytest: Ejecuta la suite de pruebas.

docker-compose up: Levanta la infraestructura completa.

###👩‍💻 Equipo de Desarrollo
Raul - Naiza - Michel - Gabriela 
 

## 📄🎯 Propósito Del Proyecto
Nuestro propósito es darle voz a los datos para salvar vidas. Somos los encargados de leer lo que los números nos dicen sobre el bienestar animal y traducirlo en acciones claras que el personal del refugio pueda ejecutar.


# Proyecto Clasificación 

## Equipo
- Gabriela Hernandez (Scrum Master)
- Raul Machaca (Product Owner)
- Michelle Gelves (Developer)
- Naizabeth Bermudez (Developer)

## Instalación y configuración

### Requisitos previos
- Python 3.x
- [uv](https://docs.astral.sh/uv/) instalado

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Mac/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Pasos

1. Clona el repositorio:
```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
```

2. Crea el entorno virtual:
```bash
uv venv
```

3. Activa el entorno virtual:

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```
**Mac/Linux o Git Bash:**
```bash
source .venv/Scripts/activate
```

4. Instala las dependencias:
```bash
uv sync
```


