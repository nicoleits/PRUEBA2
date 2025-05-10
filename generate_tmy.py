import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

# --- Función de limpieza ---
def clean_data(df):
    print("Iniciando limpieza de datos...")
    cleaning_log = []
    cleaning_log.append("Data Cleaning Summary:")
    cleaning_log.append("=====================")
    cols_to_clean = {
        'GHI': {'mark_upper_as_nan': 1250, 'interpolate': True, 'clip_lower': 0}, 
        'DNI': {'mark_upper_as_nan': 1000, 'interpolate': True, 'clip_lower': 0, 'clip_upper': None},
        'DHI': {'interpolate': True, 'clip_lower': 0, 'clip_upper': None}, 
        'TempC': {'interpolate': True, 'clip_lower': None, 'clip_upper': None},
        'Wind_mps': {'interpolate': True, 'clip_lower': 0, 'clip_upper': None}
    }
    for col, actions in cols_to_clean.items():
        if col in df.columns:
            log_entry = [f"\nColumn: {col}"]
            print(f"Limpiando {col}...")
            initial_nan = df[col].isnull().sum()
            log_entry.append(f"  Initial NaNs: {initial_nan}")
            marked_as_nan = 0
            upper_nan_limit = actions.get('mark_upper_as_nan')
            if upper_nan_limit is not None:
                outlier_mask = df[col] > upper_nan_limit
                marked_as_nan = outlier_mask.sum()
                if marked_as_nan > 0:
                    df.loc[outlier_mask, col] = np.nan
                    print(f"  Marcados {marked_as_nan} valores > {upper_nan_limit} como NaN.")
                    log_entry.append(f"  Outliers > {upper_nan_limit} marked as NaN: {marked_as_nan}")
            if actions.get('interpolate'):
                # Solo rellenar huecos de hasta 3 horas consecutivas
                df[col] = df[col].interpolate(method='linear', limit=3)
                df[col] = df[col].ffill(limit=3)
                df[col] = df[col].bfill(limit=3)
                after_limited_fill_nan = df[col].isnull().sum()
                print(f"  NaNs tras interpolación limitada: {after_limited_fill_nan}")
                log_entry.append(f"  NaNs after limited interpolation/fill: {after_limited_fill_nan}")
                # Ahora rellenar con 0 los NaN restantes
                if after_limited_fill_nan > 0:
                    df[col] = df[col].fillna(0)
                    print(f"  NaNs restantes rellenados con 0: {after_limited_fill_nan}")
                    log_entry.append(f"  Remaining NaNs filled with 0: {after_limited_fill_nan}")
                final_nan = df[col].isnull().sum()
                log_entry.append(f"  Final NaNs: {final_nan}")
            lower_limit = actions.get('clip_lower')
            if lower_limit is not None:
                negative_mask = df[col] < lower_limit
                negative_count = negative_mask.sum()
                if negative_count > 0:
                    df[col] = df[col].clip(lower=lower_limit)
                    print(f"  Corregidos {negative_count} valores < {lower_limit} en {col}.")
                    log_entry.append(f"  Values < {lower_limit} clipped: {negative_count}")
            upper_limit = actions.get('clip_upper')
            if upper_limit is not None:
                upper_mask = df[col] > upper_limit
                upper_count = upper_mask.sum()
                if upper_count > 0:
                    df[col] = df[col].clip(upper=upper_limit)
                    print(f"  Corregidos {upper_count} valores > {upper_limit} en {col}.")
                    log_entry.append(f"  Values > {upper_limit} clipped: {upper_count}")
            cleaning_log.extend(log_entry)
    print("Limpieza de datos completada.")
    cleaning_log.append("\nLimpieza de datos completada.")
    return df, cleaning_log

# --- Script principal ---
input_path = '/home/nicole/UA/NicoleTorres/solardatascience/PRUEBA2/Chile_meteo_dirty_2018_2022.csv'
output_dir = '/home/nicole/UA/NicoleTorres/solardatascience/PRUEBA2/' 
os.makedirs(output_dir, exist_ok=True)
plots_dir = os.path.join(output_dir, 'graficos_limpieza')
os.makedirs(plots_dir, exist_ok=True)

# Extraer nombre del país para el nombre del archivo limpio
match = re.search(r'/([^/_]+)_meteo', input_path)
if match:
    country_name = match.group(1)
else:
    country_name = 'cleaned'
output_clean_path = os.path.join(output_dir, f'{country_name}_meteo_clean.csv')

# Cargar y limpiar datos
df = pd.read_csv(input_path)
input_row_count = len(df)
df_cleaned, cleaning_log = clean_data(df.copy())
output_row_count = len(df_cleaned)

# Convertir 'Timestamp' a datetime una sola vez
df_cleaned['Timestamp'] = pd.to_datetime(df_cleaned['Timestamp'])

# Guardar CSV limpio
df_cleaned.to_csv(output_clean_path, index=False)
print(f"CSV limpio guardado en: {output_clean_path}")

# Guardar informe de limpieza
report_file_path = os.path.join(output_dir, 'cleaning_report.txt')
with open(report_file_path, 'w') as f:
    f.write("==============================================\n")
    f.write(f" Data Cleaning Report for {country_name} \n")
    f.write("==============================================\n")
    f.write(f"\nNúmero de filas en el CSV de entrada: {input_row_count}\n")
    f.write(f"Número de filas en el CSV limpio: {output_row_count}\n")
    f.write("\n--- Cleaning Summary ---\n")
    for line in cleaning_log:
        f.write(line + "\n")
print(f"Informe de limpieza guardado en: {report_file_path}")

# --- Informe EDA ---
eda_report_path = os.path.join(output_dir, 'eda_report.txt')
numeric_df = df_cleaned.select_dtypes(include=np.number)
with open(eda_report_path, 'w') as f:
    f.write("==============================================\n")
    f.write(f" EDA Report for {country_name} (Cleaned Data) \n")
    f.write("==============================================\n")
    f.write(f"\nNúmero de filas en el CSV limpio: {output_row_count}\n")
    f.write("\n--- Descriptive Statistics ---\n")
    f.write(df_cleaned.describe().to_string())
    f.write("\n\n--- Correlation Matrix (numeric columns only) ---\n")
    f.write(numeric_df.corr().to_string())
    f.write("\n\n--- NaN Count per Column (final, after cleaning) ---\n")
    f.write(df_cleaned.isnull().sum().to_string())
print(f"Informe EDA guardado en: {eda_report_path}")

# --- Gráficos EDA ---
# 1. Serie anual de GHI
ghi_series_path = os.path.join(plots_dir, 'ghi_annual_series.png')
try:
    plt.figure(figsize=(15, 6))
    plt.plot(df_cleaned['Timestamp'], df_cleaned['GHI'])
    plt.title('Serie Anual de GHI (Datos Limpiados)')
    plt.xlabel('Fecha')
    plt.ylabel('GHI (W/m^2)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ghi_series_path)
    plt.close()
    print(f"Gráfico guardado: {ghi_series_path}")
except Exception as e:
    print(f"Error al generar gráfico de serie anual de GHI: {e}")

# 2. Histograma de GHI
ghi_hist_path = os.path.join(plots_dir, 'ghi_histogram.png')
try:
    plt.figure(figsize=(10, 6))
    sns.histplot(df_cleaned['GHI'], kde=True, bins=50)
    plt.title('Histograma de GHI (Datos Limpiados)')
    plt.xlabel('GHI (W/m^2)')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ghi_hist_path)
    plt.close()
    print(f"Gráfico guardado: {ghi_hist_path}")
except Exception as e:
    print(f"Error al generar histograma de GHI: {e}")

# 3. Matriz de correlación
corr_matrix_path = os.path.join(plots_dir, 'correlation_matrix.png')
try:
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Matriz de Correlación (Datos Limpiados)')
    plt.tight_layout()
    plt.savefig(corr_matrix_path)
    plt.close()
    print(f"Gráfico guardado: {corr_matrix_path}")
except Exception as e:
    print(f"Error al generar matriz de correlación: {e}")

# 4. Serie de GHI para enero de un solo año
try:
    first_year = df_cleaned['Timestamp'].dt.year.min()
    january_data = df_cleaned[(df_cleaned['Timestamp'].dt.month == 1) & (df_cleaned['Timestamp'].dt.year == first_year)]
    ghi_jan_path = os.path.join(plots_dir, f'ghi_january_{first_year}.png')
    plt.figure(figsize=(15, 6))
    plt.plot(january_data['Timestamp'], january_data['GHI'])
    plt.title(f'Serie de GHI para Enero {first_year} (Datos Limpiados)')
    plt.xlabel('Fecha y hora')
    plt.ylabel('GHI (W/m^2)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ghi_jan_path)
    plt.close()
    print(f"Gráfico guardado: {ghi_jan_path}")
except Exception as e:
    print(f"Error al generar gráfico mensual de GHI: {e}")

# 5. Serie de GHI para el 21 de junio de un solo año
try:
    june21_data = df_cleaned[(df_cleaned['Timestamp'].dt.month == 6) & (df_cleaned['Timestamp'].dt.day == 21) & (df_cleaned['Timestamp'].dt.year == first_year)]
    ghi_june21_path = os.path.join(plots_dir, f'ghi_june21_{first_year}.png')
    plt.figure(figsize=(10, 5))
    plt.plot(june21_data['Timestamp'].dt.hour, june21_data['GHI'], marker='o')
    plt.title(f'Serie de GHI para el 21 de Junio {first_year} (Datos Limpiados)')
    plt.xlabel('Hora del día')
    plt.ylabel('GHI (W/m^2)')
    plt.grid(True)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(ghi_june21_path)
    plt.close()
    print(f"Gráfico guardado: {ghi_june21_path}")
except Exception as e:
    print(f"Error al generar gráfico diario de GHI: {e}") 