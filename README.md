# ☀️ Simulación y Dashboard de Planta Fotovoltaica de 50 MW

Este proyecto realiza la limpieza, análisis y simulación de datos solares horarios para tres localidades del norte de Chile (Calama, Salvador y Vallenar), utilizando modelos de PySAM y visualización interactiva con Dash. Incluye análisis de sensibilidad económico y generación de gráficos tipo tornado.

---

## 📋 Estructura del Proyecto

```
.
├── 01_limpieza_qa.ipynb         # Limpieza y análisis exploratorio de datos TMY
├── 02_simulacion_pv.ipynb       # Simulación de planta FV y análisis de sensibilidad
├── 03_dash.ipynb                # Dashboard interactivo con Dash y Plotly
├── nicole_torres.ipynb          # Notebook completo de la evaluación (flujo integral)
├── datos_limpios/               # Archivos TMY limpios y finales por localidad
├── resultados_pv/               # Resultados de simulación, gráficos y CSVs de sensibilidad
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Este archivo
```

---

## 🚀 Flujo de trabajo recomendado

### 1. Preparar el entorno
Se recomienda usar un entorno virtual (por ejemplo, con `venv` o `conda`).

### 2. Instalar dependencias
Ejecuta en la terminal:
```bash
pip install -r requirements.txt
```

### 3. Limpieza y validación de datos meteorológicos
- Utiliza el notebook `01_limpieza_qa.ipynb` o el script `limpieza.py` para transformar, limpiar y validar los archivos TMY de cada localidad.
- El proceso incluye:
  - Detección y tratamiento de valores nulos, negativos y outliers físicos en GHI, DNI, DHI.
  - Interpolación robusta de huecos pequeños y relleno jerárquico de NaNs.
  - Validación visual y estadística de la limpieza.
  - Generación de reportes EDA y gráficos comparativos.
- Los archivos limpios se guardan en `datos_limpios/`.

### 4. Simulación de la planta fotovoltaica
- Ejecuta el notebook `02_simulacion_pv.ipynb` para simular la producción horaria de una planta FV de 50 MWDC en cada localidad usando PySAM (PVWatts).
- Se calculan indicadores clave: energía anual, LCOE, VAN y se realiza análisis de sensibilidad económico.
- Los resultados y gráficos se guardan en `resultados_pv/`.

### 5. Visualización interactiva
- Abre y ejecuta el notebook `03_dash.ipynb` para lanzar el dashboard interactivo (Dash + Plotly).
- Permite explorar curvas horarias, KPIs, análisis de sensibilidad y comparar localidades de forma dinámica.

### 6. Código principal
- Abre y ejecuta el notebook `nicole_torres.ipynb` para revisar el trabajo completo.
---

## 🧩 Dependencias y entorno
- Python 3.10+
- pandas
- numpy
- matplotlib
- plotly
- dash
- pvlib
- PySAM (NREL-PySAM)
- pathlib

Instalación rápida:
```bash
pip install -r requirements.txt
```

---

## 📁 Descripción de carpetas y archivos
- `datos_limpios/`: Archivos TMY limpios y finales para cada localidad (formato esperado: columnas Year, Month, Day, Hour, Minute, GHI, DNI, DHI, etc.).
- `resultados_pv/`: Resultados de simulación, archivos CSV de sensibilidad y gráficos generados.
- `limpieza.py`: Script profesional y documentado para limpieza, validación y simulación de plantas FV.
- `nicole_torres.ipynb`: Notebook integral con todo el flujo, desde limpieza hasta dashboard y conclusiones.
- `01_limpieza_qa.ipynb`: Notebook para limpieza y análisis exploratorio de los datos solares.
- `02_simulacion_pv.ipynb`: Notebook para simulación de la planta FV y análisis de sensibilidad.
- `03_dash.ipynb`: Dashboard interactivo para visualizar datos, resultados y análisis de sensibilidad.
- `requirements.txt`: Lista de dependencias para instalación rápida.

---

## 📊 Interpretación de resultados y buenas prácticas
- **Limpieza avanzada**: Se recomienda validar visualmente los datos limpios y revisar los reportes EDA generados en `reports/`.
- **Simulación**: Los archivos TMY limpios son la entrada para la simulación PV. Asegúrate de que no existan NaNs ni outliers físicos antes de simular.
- **Análisis económico**: El LCOE y VAN se calculan considerando parámetros realistas de inversión, O&M y pérdidas. El análisis de sensibilidad permite identificar los factores más críticos.
- **Dashboard**: El dashboard permite comparar localidades, años y visualizar el impacto de cada parámetro en los KPIs.

---

## 📝 Conclusiones clave (según `nicole_torres.ipynb`)
- Salvador tiene el mejor desempeño económico (mayor VAN, bajo LCOE).
- Calama también es competitivo, con potencial de mejora mediante reducción de CapEx o pérdidas.
- Vallenar no es viable bajo las condiciones actuales, con VAN negativo y bajo rendimiento.
- El análisis de sensibilidad muestra fuerte dependencia del modelo económico al precio spot y la inversión.
- El dashboard permite explorar dinámicamente los resultados por país y año.

---

## 📌 Notas de replicabilidad
- Asegúrate de que los archivos TMY limpios estén en la carpeta `datos_limpios/` y tengan el formato esperado.
- El notebook `nicole_torres.ipynb` puede ejecutarse de principio a fin para obtener todos los resultados y gráficos necesarios para el análisis económico y energético.
- Los resultados y gráficos se guardan automáticamente en la carpeta `resultados_pv/`.

---

## 📞 Contacto
Para dudas o sugerencias, contacta a nicole.torres.silva@ua.cl.