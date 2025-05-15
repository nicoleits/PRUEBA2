☀️ Simulación y Dashboard de Planta Fotovoltaica de 50 MW
Este proyecto realiza la limpieza, análisis y simulación de datos solares horarios para tres localidades del norte de Chile (Calama, Salvador y Vallenar), utilizando modelos de PySAM y visualización interactiva con Dash. Incluye análisis de sensibilidad económico y generación de gráficos tipo tornado.

📋 Estructura del Proyecto
.
├── 01_limpieza_qa.ipynb         # Limpieza y análisis exploratorio de datos TMY
├── 02_simulacion_pv.ipynb       # Simulación de planta FV y análisis de sensibilidad
├── 03_dash.ipynb                # Dashboard interactivo con Dash y Plotly
├── datos_limpios/               # Archivos TMY limpios y finales por localidad
├── resultados_pv/               # Resultados de simulación, gráficos y CSVs de sensibilidad
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Este archivo

🚀 Cómo ejecutar el trabajo
1. Preparar el entorno
Se recomienda usar un entorno virtual (por ejemplo, con venv o conda).
2. Instalar dependencias
Ejecuta en la terminal:    pip install -r requirements.txt

3. Procesar y limpiar los datos
Abre y ejecuta el notebook 01_limpieza_qa.ipynb.
Esto generará archivos TMY limpios en la carpeta datos_limpios/.
4. Simular la planta fotovoltaica
Abre y ejecuta el notebook 02_simulacion_pv.ipynb.
Se generarán resultados de simulación y análisis de sensibilidad en resultados_pv/.
5. Visualizar resultados en el dashboard
Abre y ejecuta el notebook 03_dash.ipynb.
Se abrirá un dashboard interactivo en Jupyter para explorar los datos y resultados.

🧩 Dependencias y entorno
-   Python 3.10+
-   pandas
-   numpy
-   matplotlib
-   plotly
-   dash
-   pvlib
-   PySAM
-   pathlib

Instalación rápida: pip install -r requirements.txt
-   pandas>=1.5
-   numpy>=1.23
-   matplotlib>=3.5
-   plotly>=5.10
-   dash>=2.7
-   pvlib>=0.9
-   NREL-PySAM>=3.0

📁 Descripción de carpetas y archivos
datos_limpios/: Contiene los archivos TMY limpios y finales para cada localidad.
resultados_pv/: Incluye los resultados de simulación, archivos CSV de sensibilidad y gráficos generados.
01_limpieza_qa.ipynb: Notebook para limpieza y análisis exploratorio de los datos solares.
02_simulacion_pv.ipynb: Notebook para simulación de la planta FV y análisis de sensibilidad.
03_dash.ipynb: Dashboard interactivo para visualizar datos, resultados y análisis de sensibilidad.
requirements.txt: Lista de dependencias para instalación rápida.

📞 Contacto
Para dudas o sugerencias, contacta a nicole.torres.silva@ua.cl.