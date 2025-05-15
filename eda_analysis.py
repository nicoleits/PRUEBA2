import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import pvlib
import os
from pvlib.solarposition import get_solarposition

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
    """
    # Crear directorio para informes si no existe
    Path(output_dir).mkdir(exist_ok=True)
    
    # Crear archivo de informe
    report_path = Path(output_dir) / f"{location.lower()}_eda_report.txt"
    
    # Definir límites de outliers
    outlier_limits = {
        'GHI': 1200,  # Actualizado para coincidir con limpiar_TMY_completo
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
        f.write("- GHI > 1200 W/m² o < 0\n")  # Actualizado
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
        f.write("   - GHI: 1200 W/m²\n")  # Actualizado
        f.write("   - DNI: 1300 W/m²\n")
        f.write("   - DHI: 400 W/m²\n")
        f.write("3. Considerar la interpolación solo para secuencias cortas de NaN (≤ 4 horas)\n")
        f.write("4. Revisar la calidad de los datos en los meses con mayor concentración de outliers\n")
        f.write("5. Documentar el proceso de limpieza y las decisiones tomadas para el manejo de outliers\n")

def analisis_estacional(df, location):
    """
    Realiza un análisis estacional de los datos solares.
    """
    print(f"\n=== ANÁLISIS ESTACIONAL - {location} ===")
    print("=" * 50)
    
    # Definir estaciones
    estaciones = {
        'Verano': [12, 1, 2],
        'Otoño': [3, 4, 5],
        'Invierno': [6, 7, 8],
        'Primavera': [9, 10, 11]
    }
    
    # Crear DataFrame para almacenar estadísticas estacionales
    stats_estacionales = pd.DataFrame(index=estaciones.keys(), 
                                    columns=['GHI Promedio', 'DNI Promedio', 'DHI Promedio',
                                            'GHI Máximo', 'DNI Máximo', 'DHI Máximo',
                                            'Horas de Sol', 'Energía Total'])
    
    # Calcular estadísticas por estación
    for estacion, meses in estaciones.items():
        df_estacion = df[df['Month'].isin(meses)]
        
        # Estadísticas básicas
        stats_estacionales.loc[estacion, 'GHI Promedio'] = df_estacion['GHI'].mean()
        stats_estacionales.loc[estacion, 'DNI Promedio'] = df_estacion['DNI'].mean()
        stats_estacionales.loc[estacion, 'DHI Promedio'] = df_estacion['DHI'].mean()
        
        stats_estacionales.loc[estacion, 'GHI Máximo'] = df_estacion['GHI'].max()
        stats_estacionales.loc[estacion, 'DNI Máximo'] = df_estacion['DNI'].max()
        stats_estacionales.loc[estacion, 'DHI Máximo'] = df_estacion['DHI'].max()
        
        # Horas de sol y energía
        stats_estacionales.loc[estacion, 'Horas de Sol'] = len(df_estacion[df_estacion['GHI'] > 0])
        stats_estacionales.loc[estacion, 'Energía Total'] = df_estacion['GHI'].sum() / 1000
    
    # Mostrar estadísticas estacionales
    print("\nEstadísticas por Estación:")
    print(stats_estacionales.round(2))

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

def limpiar_TMY_completo(archivo_entrada, archivo_salida, max_ghi=1400, max_dni=1300, max_dhi=400, interp_limit=6, location=None):
    """
    Limpia GHI, DNI y DHI de un archivo TMY artificial y guarda un solo archivo limpio con metadatos.
    """
    # Leer metadatos
    with open(archivo_entrada, 'r', encoding='utf-8') as f:
        metadatos = [next(f) for _ in range(2)]
    # Leer datos desde la tercera línea
    df = pd.read_csv(archivo_entrada, skiprows=2)
    # Crear columna datetime y usar como índice
    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df = df.set_index("datetime")
    
    # Ajustar límites según la ubicación
    if location == "Vallenar":
        max_ghi = 1200
        ghi_high_season = 1100
    else:
        max_ghi = 1400
        ghi_high_season = 1200
    
    # Limpiar GHI con límites dependientes del mes y ubicación
    df["GHI"] = df["GHI"].mask(
        ((df["Month"].between(3, 10)) & (df["GHI"] >= ghi_high_season)) |  # Límite para meses de alta radiación
        ((~df["Month"].between(3, 10)) & (df["GHI"] > max_ghi)) |  # Límite general
        (df["GHI"] < 0)
    )
    df["GHI"] = df["GHI"].interpolate(method='linear', limit=interp_limit, limit_direction='both')
    
    # Limpiar DNI
    df["DNI"] = df["DNI"].mask((df["DNI"] < 0) | (df["DNI"] > max_dni))
    df["DNI"] = df["DNI"].interpolate(method='linear', limit=interp_limit, limit_direction='both')
    
    # Limpiar DHI con chequeos físicos avanzados
    # Calcular posición solar para validación física
    lat = -26.2533 if location == "Salvador" else -22.4661 if location == "Calama" else -28.5766
    lon = -69.0522 if location == "Salvador" else -68.9244 if location == "Calama" else -70.7601
    solar_position = get_solarposition(df.index, latitude=lat, longitude=lon)
    cos_zenith = np.cos(np.radians(solar_position["zenith"]))
    dhi_est = (df["GHI"] - df["DNI"] * cos_zenith).clip(lower=0)

    cond_invalid_dhi = (
        (df["DHI"] < 0) |
        (df["DHI"] > max_dhi) |
        (df["DHI"] > df["GHI"]) |
        (df["DHI"] > dhi_est + 20) |
        (df["DHI"] > 0.95 * df["GHI"]) |
        ((solar_position["zenith"] > 90) & (df["DHI"] > 5))
    )
    df.loc[cond_invalid_dhi, "DHI"] = np.nan

    # Interpolación robusta para DHI
    def interpolar_robusto(serie, limit):
        nan_groups = serie.isna().astype(int).groupby(serie.notna().astype(int).cumsum()).sum()
        if (nan_groups > limit).any():
            mask = serie.isna()
            for idx, size in nan_groups[nan_groups > limit].items():
                mask[mask.groupby(mask.cumsum()).ngroup() == idx] = False
            serie_interp = serie.interpolate(method='linear', limit=limit, limit_direction='both')
            serie[mask] = serie_interp[mask]
            return serie
        else:
            return serie.interpolate(method='linear', limit=limit, limit_direction='both')

    df["DHI"] = interpolar_robusto(df["DHI"], interp_limit)
    
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
    
    # --- NUEVOS OUTLIERS ESPECÍFICOS POR UBICACIÓN Y MES ---
    if location == "Vallenar":
        # GHI > 1000 entre abril y agosto
        df["GHI"] = df["GHI"].mask((df["Month"].between(4, 8)) & (df["GHI"] > 1000))
        # DNI > 1150 todo el año
        df["DNI"] = df["DNI"].mask(df["DNI"] > 1150)
    elif location == "Calama":
        # GHI > 1000 entre mayo y julio
        df["GHI"] = df["GHI"].mask((df["Month"].between(5, 7)) & (df["GHI"] > 1000))
        # DNI > 1200 todo el año
        df["DNI"] = df["DNI"].mask(df["DNI"] > 1200)
    elif location == "Salvador":
        # GHI > 1100 entre abril y agosto
        df["GHI"] = df["GHI"].mask((df["Month"].between(4, 8)) & (df["GHI"] > 1100))
    # --- FIN NUEVOS OUTLIERS ---
    
    # Guardar nuevo archivo con metadatos originales
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        f.writelines(metadatos)
        f.write(",".join(df.columns) + "\n")
        df.to_csv(f, index=False, header=False)
    print(f"✅ Archivo limpio guardado en: {archivo_salida}")

def generar_resumen_comparativo(dfs, locations):
    """
    Genera un resumen comparativo del EDA para todas las ubicaciones.
    """
    print("\n=== RESUMEN COMPARATIVO DE DATOS SOLARES ===")
    print("=" * 50)
    
    # Análisis de valores faltantes
    print("\nAnálisis de Valores Faltantes:")
    for df, location in zip(dfs, locations):
        nan_counts = df[['GHI', 'DNI', 'DHI']].isna().sum()
        print(f"\n{location}:")
        for col in ['GHI', 'DNI', 'DHI']:
            print(f"  {col}: {nan_counts[col]} valores faltantes ({(nan_counts[col]/len(df))*100:.2f}%)")
    
    # Análisis de outliers
    print("\nAnálisis de Outliers:")
    outlier_limits = {'GHI': 1200, 'DNI': 1300, 'DHI': 400}  # Actualizado
    for df, location in zip(dfs, locations):
        print(f"\n{location}:")
        for col in ['GHI', 'DNI', 'DHI']:
            neg_outliers = len(df[df[col] < 0])
            high_outliers = len(df[df[col] > outlier_limits[col]])
            total_outliers = neg_outliers + high_outliers
            print(f"  {col}: {total_outliers} outliers ({(total_outliers/len(df))*100:.2f}%)")
            if total_outliers > 0:
                print(f"    - Negativos: {neg_outliers}")
                print(f"    - Altos: {high_outliers}")

def reconstruir_TMY_con_fechas_originales(archivo_corrupto, archivo_limpio, archivo_salida):
    """
    Genera un nuevo archivo TMY limpio con los años y meses originales del archivo fuente,
    pero con los valores limpios. Mantiene los metadatos y el encabezado.
    """
    # Leer datos limpios
    df_limpio = pd.read_csv(archivo_limpio, skiprows=2)
    # Leer datos originales (sin metadatos)
    df_original = pd.read_csv(archivo_corrupto)
    # Reemplazar columnas de fecha por las originales
    for col in ['Year', 'Month', 'Day', 'Hour', 'Minute']:
        if col in df_original.columns and col in df_limpio.columns:
            df_limpio[col] = df_original[col].values
    # Reordenar columnas para mantener el formato
    columnas_fecha = ['Year', 'Month', 'Day', 'Hour', 'Minute']
    columnas_finales = columnas_fecha + [col for col in df_limpio.columns if col not in columnas_fecha]
    df_limpio = df_limpio[columnas_finales]
    # Leer metadatos (primeras 3 líneas del archivo limpio)
    with open(archivo_limpio, 'r', encoding='utf-8') as f:
        metadatos = [next(f) for _ in range(3)]
    # Guardar el nuevo archivo
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        f.writelines(metadatos)
        df_limpio.to_csv(f, index=False, header=False)
    print(f"✅ Archivo TMY limpio con fechas originales guardado en: {archivo_salida}")

def main():
    print("Iniciando procesamiento...")
    # Definir los metadatos para cada sitio
    metadatos_salvador = {
        "Source": "ExpSolar", "Location ID": "00001", "City": "Salvador", "State": "Atacama",
        "Country": "Chile", "Latitude": -26.2533, "Longitude": -69.0522, "Time Zone": -4, "Elevation": 2280
    }
    metadatos_calama = {
        "Source": "ExpSolar", "Location ID": "00002", "City": "Calama", "State": "Antofagasta",
        "Country": "Chile", "Latitude": -22.4661, "Longitude": -68.9244, "Time Zone": -4, "Elevation": 2260
    }
    metadatos_vallenar = {
        "Source": "ExpSolar", "Location ID": "00003", "City": "Vallenar", "State": "Atacama",
        "Country": "Chile", "Latitude": -28.5766, "Longitude": -70.7601, "Time Zone": -4, "Elevation": 441
    }
    # Rutas de archivos
    path_salvador = Path("salvador_corrupted.csv")
    path_calama = Path("calama_corrupted.csv")
    path_vallenar = Path("Vallenar_corrupted.csv")
    # Salidas
    output_salvador = Path("salvador_TMY_artificial.csv")
    output_calama = Path("calama_TMY_artificial.csv")
    output_vallenar = Path("vallenar_TMY_artificial.csv")
    output_salvador_limpio = Path("salvador_TMY_limpio.csv")
    output_calama_limpio = Path("calama_TMY_limpio.csv")
    output_vallenar_limpio = Path("vallenar_TMY_limpio.csv")
    
    # Lista para almacenar los DataFrames limpios
    dfs_limpios = []
    locations = []
    
    # Procesar cada archivo
    for location, (input_path, output_path, output_limpio, metadata) in [
        ("Salvador", (path_salvador, output_salvador, output_salvador_limpio, metadatos_salvador)),
        ("Calama", (path_calama, output_calama, output_calama_limpio, metadatos_calama)),
        ("Vallenar", (path_vallenar, output_vallenar, output_vallenar_limpio, metadatos_vallenar))
    ]:
        try:
            print(f"\nProcesando {location}...")
            print(f"Leyendo archivo: {input_path}")
            df = transformar_a_tmy_con_metadatos(input_path, output_path, metadata)
            print(f"Transformación completada. Limpiando datos...")
            limpiar_TMY_completo(str(output_path), str(output_limpio), location=location)
            print(f"Limpieza completada. Eliminando archivos temporales...")
            for suffix in ["_GHI_limpio.csv", "_DNI_limpio.csv", "_DHI_limpio.csv"]:
                try:
                    os.remove(str(output_path).replace("_TMY_artificial.csv", suffix))
                except FileNotFoundError:
                    pass
            print(f"Leyendo archivo limpio...")
            df_limpio = pd.read_csv(output_limpio, skiprows=2)
            dfs_limpios.append(df_limpio)
            locations.append(location)
            print(f"Generando gráficos...")
            plot_tmy_data(df_limpio, location)
            print(f"Generando informe EDA...")
            generate_eda_report(df_limpio, location)
            print(f"Realizando análisis estacional...")
            analisis_estacional(df_limpio, location)
            print(f"Eliminando archivo temporal...")
            try:
                os.remove(str(output_path))
            except FileNotFoundError:
                pass
            print(f"Procesamiento de {location} completado.")
        except Exception as e:
            print(f"Error procesando {location}: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    # Generar resumen comparativo
    if len(dfs_limpios) == 3:  # Solo si se procesaron los tres archivos correctamente
        print("\nGenerando resumen comparativo...")
        generar_resumen_comparativo(dfs_limpios, locations)

        # Reconstruir TMY limpio con fechas originales para cada localidad
        print("\nReconstruyendo archivos TMY limpios con fechas originales...")
        reconstruir_TMY_con_fechas_originales("salvador_corrupted.csv", "salvador_TMY_limpio.csv", "salvador_TMY_limpio_originales.csv")
        reconstruir_TMY_con_fechas_originales("calama_corrupted.csv", "calama_TMY_limpio.csv", "calama_TMY_limpio_originales.csv")
        reconstruir_TMY_con_fechas_originales("Vallenar_corrupted.csv", "vallenar_TMY_limpio.csv", "vallenar_TMY_limpio_originales.csv")
    else:
        print(f"\nNo se pudo generar el resumen comparativo. Se procesaron {len(dfs_limpios)} de 3 archivos.")

if __name__ == "__main__":
    main() 