import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import pvlib

# Configurar pandas para evitar downcasting silencioso
pd.set_option('future.no_silent_downcasting', True)

def transformar_a_tmy_con_metadatos(csv_path, output_path, metadata_dict):
    """
    Transforma un archivo CSV con datos horarios en un TMY artificial con metadatos en el encabezado.
    """
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)

    # Eliminar columna 'datetime' si existe
    if 'datetime' in df.columns:
        df = df.drop(columns=['datetime'])

    # Ordenar cronológicamente y recortar o completar a 8760 filas
    df = df.sort_values(by=["Month", "Day", "Hour", "Minute"])
    df = df.reset_index(drop=True)
    df = df.iloc[:8760]  # en caso de que tenga más filas

    # Asignar un año artificial constante
    df['Year'] = 1990

    # Reordenar columnas: primero las de fecha
    columnas_fecha = ['Year', 'Month', 'Day', 'Hour', 'Minute']
    otras = [c for c in df.columns if c not in columnas_fecha]
    df = df[columnas_fecha + otras]

    # Escribir archivo con las tres primeras líneas de metadatos
    with open(output_path, "w", encoding="utf-8") as f:
        # Línea 1: encabezados de metadatos
        f.write("Source,Location ID,City,State,Country,Latitude,Longitude,Time Zone,Elevation\n")
        # Línea 2: valores de metadatos
        f.write(",".join(str(metadata_dict[k]) for k in [
            "Source", "Location ID", "City", "State", "Country",
            "Latitude", "Longitude", "Time Zone", "Elevation"
        ]) + "\n")
        # Línea 3: encabezado de columnas de datos
        f.write(",".join(df.columns) + "\n")
        # Resto de datos
        df.to_csv(f, index=False, header=False)
    
    return df

def plot_tmy_data(df, location, output_dir="plots", is_clean=False):
    """
    Genera gráficos de los datos TMY artificiales.
    
    Args:
        df: DataFrame con los datos TMY
        location: Nombre de la ubicación
        output_dir: Directorio donde guardar los gráficos
        is_clean: Booleano que indica si los datos están limpios
    """
    # Crear directorio para gráficos si no existe
    Path(output_dir).mkdir(exist_ok=True)
    
    # Configurar estilo de gráficos
    plt.style.use('default')
    
    # Crear figura con tres subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
    title_suffix = " (Datos Limpios)" if is_clean else ""
    fig.suptitle(f'Radiación Solar por Hora - {location}{title_suffix}', fontsize=16, y=0.95)
    
    # Crear fechas para el eje x
    dates = [datetime(1990, int(row['Month']), int(row['Day']), int(row['Hour'])) 
             for _, row in df.iterrows()]
    
    # Graficar cada componente en su propio subplot
    ax1.plot(dates, df['GHI'], 'b-', linewidth=1, alpha=0.7)
    ax1.set_title('Radiación Global Horizontal (GHI)', fontsize=12)
    ax1.set_ylabel('GHI (W/m²)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1300)  # Establecer límite superior para GHI
    
    ax2.plot(dates, df['DNI'], 'r-', linewidth=1, alpha=0.7)
    ax2.set_title('Radiación Normal Directa (DNI)', fontsize=12)
    ax2.set_ylabel('DNI (W/m²)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1300)  # Establecer límite superior para DNI
    
    ax3.plot(dates, df['DHI'], 'g-', linewidth=1, alpha=0.7)
    ax3.set_title('Radiación Horizontal Difusa (DHI)', fontsize=12)
    ax3.set_ylabel('DHI (W/m²)', fontsize=10)
    ax3.set_xlabel('Mes', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 400)  # Establecer límite superior para DHI
    
    # Configurar formato del eje x para todos los subplots
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Ajustar layout y guardar
    plt.tight_layout()
    suffix = "_clean" if is_clean else ""
    plt.savefig(f'{output_dir}/{location.lower()}_tmy_plots{suffix}.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def generate_eda_report(df, location, output_dir="reports"):
    """
    Genera un informe EDA con análisis de valores faltantes y outliers.
    Outliers definidos como:
    - GHI > 1300 W/m² o < 0
    - DNI > 1300 W/m² o < 0
    - DHI > 400 W/m² o < 0
    
    Args:
        df: DataFrame con los datos
        location: Nombre de la ubicación
        output_dir: Directorio donde guardar el informe
    """
    # Crear directorio para informes si no existe
    Path(output_dir).mkdir(exist_ok=True)
    
    # Crear archivo de informe
    report_path = Path(output_dir) / f"{location.lower()}_eda_report.txt"
    
    # Definir límites de outliers
    outlier_limits = {
        'GHI': 1300,
        'DNI': 1300,
        'DHI': 400
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Informe EDA - {location} ===\n\n")
        
        # 1. Información general
        f.write("1. INFORMACIÓN GENERAL\n")
        f.write("-" * 50 + "\n")
        f.write(f"Número total de registros: {len(df)}\n")
        f.write(f"Período: {df['Month'].min()}/{df['Day'].min()} - {df['Month'].max()}/{df['Day'].max()}\n\n")
        
        # 2. Análisis de valores faltantes
        f.write("2. ANÁLISIS DE VALORES FALTANTES\n")
        f.write("-" * 50 + "\n")
        nan_counts = df[['GHI', 'DNI', 'DHI']].isna().sum()
        nan_percentages = (nan_counts / len(df)) * 100
        
        for col in ['GHI', 'DNI', 'DHI']:
            f.write(f"{col}:\n")
            f.write(f"  - Número de valores faltantes: {nan_counts[col]}\n")
            f.write(f"  - Porcentaje de valores faltantes: {nan_percentages[col]:.2f}%\n")
            
            # Análisis de secuencias de NaN
            if nan_counts[col] > 0:
                nan_sequences = df[col].isna().astype(int).groupby(
                    (df[col].isna().astype(int).diff() != 0).cumsum()
                ).cumsum()
                max_consecutive = nan_sequences.max()
                f.write(f"  - Máxima secuencia de NaN consecutivos: {max_consecutive}\n")
        f.write("\n")
        
        # 3. Análisis de Outliers
        f.write("3. ANÁLISIS DE OUTLIERS\n")
        f.write("-" * 50 + "\n")
        f.write("Criterios de outliers:\n")
        f.write("- GHI > 1300 W/m² o < 0\n")
        f.write("- DNI > 1300 W/m² o < 0\n")
        f.write("- DHI > 400 W/m² o < 0\n\n")
        
        for col in ['GHI', 'DNI', 'DHI']:
            # Identificar outliers (valores negativos o mayores al límite)
            neg_outliers = df[df[col] < 0][col]
            high_outliers = df[df[col] > outlier_limits[col]][col]
            total_outliers = len(neg_outliers) + len(high_outliers)
            
            f.write(f"{col}:\n")
            f.write(f"  - Número total de outliers: {total_outliers}\n")
            f.write(f"  - Porcentaje de outliers: {(total_outliers/len(df))*100:.2f}%\n")
            
            if len(neg_outliers) > 0:
                f.write(f"  - Outliers negativos: {len(neg_outliers)} ({len(neg_outliers)/total_outliers*100:.2f}% del total de outliers)\n")
                f.write(f"    * Valor mínimo: {neg_outliers.min():.2f} W/m²\n")
                f.write(f"    * Valor máximo negativo: {neg_outliers.max():.2f} W/m²\n")
                f.write(f"    * Media de outliers negativos: {neg_outliers.mean():.2f} W/m²\n")
            
            if len(high_outliers) > 0:
                f.write(f"  - Outliers altos: {len(high_outliers)} ({len(high_outliers)/total_outliers*100:.2f}% del total de outliers)\n")
                f.write(f"    * Valor mínimo: {high_outliers.min():.2f} W/m²\n")
                f.write(f"    * Valor máximo: {high_outliers.max():.2f} W/m²\n")
                f.write(f"    * Media de outliers altos: {high_outliers.mean():.2f} W/m²\n")
            
            # Análisis temporal de outliers
            if total_outliers > 0:
                f.write("  - Distribución temporal de outliers:\n")
                for month in range(1, 13):
                    month_outliers = len(df[(df['Month'] == month) & 
                                          ((df[col] < 0) | (df[col] > outlier_limits[col]))])
                    if month_outliers > 0:
                        f.write(f"    * Mes {month}: {month_outliers} outliers\n")
        f.write("\n")
        
        # 4. Estadísticas descriptivas (excluyendo outliers)
        f.write("4. ESTADÍSTICAS DESCRIPTIVAS (EXCLUYENDO OUTLIERS)\n")
        f.write("-" * 50 + "\n")
        for col in ['GHI', 'DNI', 'DHI']:
            f.write(f"{col}:\n")
            # Filtrar valores válidos (no outliers)
            valid_data = df[(df[col] >= 0) & (df[col] <= outlier_limits[col])][col]
            stats = valid_data.describe()
            f.write(f"  - Media: {stats['mean']:.2f} W/m²\n")
            f.write(f"  - Desviación estándar: {stats['std']:.2f} W/m²\n")
            f.write(f"  - Mínimo: {stats['min']:.2f} W/m²\n")
            f.write(f"  - Máximo: {stats['max']:.2f} W/m²\n")
            f.write(f"  - Mediana: {stats['50%']:.2f} W/m²\n")
            f.write(f"  - Q1 (25%): {stats['25%']:.2f} W/m²\n")
            f.write(f"  - Q3 (75%): {stats['75%']:.2f} W/m²\n")
        f.write("\n")
        
        # 5. Recomendaciones
        f.write("5. RECOMENDACIONES\n")
        f.write("-" * 50 + "\n")
        for col in ['GHI', 'DNI', 'DHI']:
            f.write(f"{col}:\n")
            if nan_counts[col] > 0:
                f.write(f"  - Considerar interpolación para {nan_counts[col]} valores faltantes ")
                if max_consecutive > 4:
                    f.write(f"(¡Alerta! Hay secuencias de hasta {max_consecutive} NaN consecutivos)\n")
                else:
                    f.write("(secuencias cortas, adecuadas para interpolación)\n")
            
            neg_count = len(df[df[col] < 0])
            high_count = len(df[df[col] > outlier_limits[col]])
            if neg_count > 0 or high_count > 0:
                f.write(f"  - Reemplazar {neg_count + high_count} outliers:\n")
                if neg_count > 0:
                    f.write(f"    * {neg_count} valores negativos con 0\n")
                if high_count > 0:
                    f.write(f"    * {high_count} valores > {outlier_limits[col]} W/m² con {outlier_limits[col]} W/m²\n")
                f.write(f"  - Revisar la calidad de los datos en los meses con mayor concentración de outliers\n")
        
        f.write("\nRecomendaciones generales:\n")
        f.write("1. Reemplazar todos los valores negativos con 0\n")
        f.write("2. Limitar los valores máximos a los umbrales físicos:\n")
        f.write("   - GHI y DNI: 1300 W/m²\n")
        f.write("   - DHI: 400 W/m²\n")
        f.write("3. Considerar la interpolación solo para secuencias cortas de NaN (≤ 4 horas)\n")
        f.write("4. Revisar la calidad de los datos en los meses con mayor concentración de outliers\n")
        f.write("5. Documentar el proceso de limpieza y las decisiones tomadas para el manejo de outliers\n")
    
    print(f"Informe EDA guardado como '{report_path}'")

def marcar_outliers_nan(df):
    """
    Marca como NaN los valores físicamente inválidos en GHI, DNI y DHI.
    - GHI < 0 o GHI > 1400
    - DNI < 0 o DNI > 1300
    - DHI < 0 o DHI > 400
    """
    df = df.copy()
    df['GHI'] = df['GHI'].mask((df['GHI'] < 0) | (df['GHI'] > 1400))
    df['DNI'] = df['DNI'].mask((df['DNI'] < 0) | (df['DNI'] > 1300))
    df['DHI'] = df['DHI'].mask((df['DHI'] < 0) | (df['DHI'] > 400))
    return df

def limpiar_GHI_tmy(archivo_entrada, archivo_salida, max_ghi=1400, interp_limit=6):
    """
    Lee un archivo TMY con metadatos, limpia la columna GHI y guarda un nuevo archivo limpio.
    """
    # Leer metadatos
    with open(archivo_entrada, 'r', encoding='utf-8') as f:
        metadatos = [next(f) for _ in range(2)]

    # Leer datos desde la tercera línea
    df = pd.read_csv(archivo_entrada, skiprows=2)

    # Crear columna datetime y usar como índice
    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df = df.set_index("datetime")

    # Paso 1: marcar como NaN los valores físicamente inválidos
    df["GHI"] = df["GHI"].mask((df["GHI"] < 0) | (df["GHI"] > max_ghi))

    # Paso 2: interpolación limitada
    df["GHI"] = df["GHI"].interpolate(method='linear', limit=interp_limit, limit_direction='both')

    # Restaurar columnas separadas
    df = df.reset_index()
    df["Year"] = df["datetime"].dt.year
    df["Month"] = df["datetime"].dt.month
    df["Day"] = df["datetime"].dt.day
    df["Hour"] = df["datetime"].dt.hour
    df["Minute"] = df["datetime"].dt.minute

    # Reordenar columnas
    columnas_fecha = ["Year", "Month", "Day", "Hour", "Minute"]
    columnas_finales = columnas_fecha + [col for col in df.columns if col not in columnas_fecha + ["datetime"]]
    df = df[columnas_finales]

    # Guardar nuevo archivo con metadatos originales
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        f.writelines(metadatos)  # líneas 1 y 2
        f.write(",".join(df.columns) + "\n")
        df.to_csv(f, index=False, header=False)

    print(f"✅ GHI limpio guardado en: {archivo_salida}")

def limpiar_DNI_tmy(archivo_entrada, archivo_salida, max_dni=1300, interp_limit=6):
    """
    Lee un archivo TMY con metadatos, limpia la columna DNI y guarda un nuevo archivo limpio.
    """
    # Leer metadatos
    with open(archivo_entrada, 'r', encoding='utf-8') as f:
        metadatos = [next(f) for _ in range(2)]

    # Leer datos desde la tercera línea
    df = pd.read_csv(archivo_entrada, skiprows=2)

    # Crear columna datetime y usar como índice
    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df = df.set_index("datetime")

    # Paso 1: marcar como NaN los valores físicamente inválidos
    df["DNI"] = df["DNI"].mask((df["DNI"] < 0) | (df["DNI"] > max_dni))

    # Paso 2: interpolación limitada
    df["DNI"] = df["DNI"].interpolate(method='linear', limit=interp_limit, limit_direction='both')

    # Restaurar columnas separadas
    df = df.reset_index()
    df["Year"] = df["datetime"].dt.year
    df["Month"] = df["datetime"].dt.month
    df["Day"] = df["datetime"].dt.day
    df["Hour"] = df["datetime"].dt.hour
    df["Minute"] = df["datetime"].dt.minute

    # Reordenar columnas
    columnas_fecha = ["Year", "Month", "Day", "Hour", "Minute"]
    columnas_finales = columnas_fecha + [col for col in df.columns if col not in columnas_fecha + ["datetime"]]
    df = df[columnas_finales]

    # Guardar nuevo archivo con metadatos originales
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        f.writelines(metadatos)  # líneas 1 y 2
        f.write(",".join(df.columns) + "\n")
        df.to_csv(f, index=False, header=False)

    print(f"✅ DNI limpio guardado en: {archivo_salida}")

def limpiar_DHI_tmy(archivo_entrada, archivo_salida, max_dhi=400, interp_limit=6):
    """
    Lee un archivo TMY con metadatos, limpia la columna DHI y guarda un nuevo archivo limpio.
    """
    # Leer metadatos
    with open(archivo_entrada, 'r', encoding='utf-8') as f:
        metadatos = [next(f) for _ in range(2)]

    # Leer datos desde la tercera línea
    df = pd.read_csv(archivo_entrada, skiprows=2)

    # Crear columna datetime y usar como índice
    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df = df.set_index("datetime")

    # Paso 1: marcar como NaN los valores físicamente inválidos
    df["DHI"] = df["DHI"].mask((df["DHI"] < 0) | (df["DHI"] > max_dhi))

    # Paso 2: interpolación limitada
    df["DHI"] = df["DHI"].interpolate(method='linear', limit=interp_limit, limit_direction='both')

    # Restaurar columnas separadas
    df = df.reset_index()
    df["Year"] = df["datetime"].dt.year
    df["Month"] = df["datetime"].dt.month
    df["Day"] = df["datetime"].dt.day
    df["Hour"] = df["datetime"].dt.hour
    df["Minute"] = df["datetime"].dt.minute

    # Reordenar columnas
    columnas_fecha = ["Year", "Month", "Day", "Hour", "Minute"]
    columnas_finales = columnas_fecha + [col for col in df.columns if col not in columnas_fecha + ["datetime"]]
    df = df[columnas_finales]

    # Guardar nuevo archivo con metadatos originales
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        f.writelines(metadatos)
        f.write(",".join(df.columns) + "\n")
        df.to_csv(f, index=False, header=False)

def main():
    # Definir los metadatos para cada sitio
    metadatos_salvador = {
        "Source": "ExpSolar", "Location ID": "00001", "City": "Salvador", "State": "Atacama",
        "Country": "Chile", "Latitude": -26.2533, "Longitude": -69.0522, "Time Zone": -4, "Elevation": 2280
    }

    metadatos_calama = {
        "Source": "ExpSolar", "Location ID": "00002", "City": "Calama", "State": "Antofagasta",
        "Country": "Chile", "Latitude": -22.4661, "Longitude": -68.9244, "Time Zone": -4, "Elevation": 2260
    }

    # Rutas de archivos
    path_salvador = Path("salvador_corrupted.csv")
    path_calama = Path("calama_corrupted.csv")

    # Salidas
    output_salvador = Path("salvador_TMY_artificial.csv")
    output_calama = Path("calama_TMY_artificial.csv")
    output_salvador_ghi_limpio = Path("salvador_TMY_GHI_limpio.csv")
    output_calama_ghi_limpio = Path("calama_TMY_GHI_limpio.csv")
    output_salvador_dni_limpio = Path("salvador_TMY_DNI_limpio.csv")
    output_calama_dni_limpio = Path("calama_TMY_DNI_limpio.csv")

    # Procesar cada archivo
    for location, (input_path, output_path, output_ghi_limpio, output_dni_limpio, metadata) in [
        ("Salvador", (path_salvador, output_salvador, output_salvador_ghi_limpio, output_salvador_dni_limpio, metadatos_salvador)),
        ("Calama", (path_calama, output_calama, output_calama_ghi_limpio, output_calama_dni_limpio, metadatos_calama))
    ]:
        try:
            print(f"\nProcesando {location}...")
            # Transformar a TMY artificial
            df = transformar_a_tmy_con_metadatos(input_path, output_path, metadata)
            print(f"Archivo TMY artificial guardado como '{output_path}'")
            # Limpiar GHI y DNI y guardar archivos limpios
            limpiar_GHI_tmy(str(output_path), str(output_ghi_limpio))
            limpiar_DNI_tmy(str(output_path), str(output_dni_limpio))
            # Leer el archivo TMY original (sucio)
            df_sucio = pd.read_csv(output_path, skiprows=2)
            # Leer las columnas limpias
            df_ghi_limpio = pd.read_csv(output_ghi_limpio, skiprows=2)["GHI"]
            df_dni_limpio = pd.read_csv(output_dni_limpio, skiprows=2)["DNI"]
            # Reemplazar en el DataFrame sucio
            df_sucio["GHI"] = df_ghi_limpio
            df_sucio["DNI"] = df_dni_limpio
            # Graficar y EDA con datos sucios pero GHI y DNI limpios
            plot_tmy_data(df_sucio, location)
            print(f"Gráficos originales guardados en 'plots/{location.lower()}_tmy_plots.png'")
            generate_eda_report(df_sucio, location)
        except Exception as e:
            print(f"Error procesando {location}: {str(e)}")

if __name__ == "__main__":
    main() 