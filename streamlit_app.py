import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(layout="wide") # Use wide layout for better dashboard feel

# --- Configuration and Data Loading ---
DATOS_LIMPIOS_DIR = Path("datos_limpios")
RESULTADOS_PV_DIR = Path("resultados_pv")

@st.cache_data # Cache data loading for performance
def load_data():
    df_hourly = pd.read_csv(RESULTADOS_PV_DIR / "pv_simulation_results_hourly.csv", parse_dates=["datetime"])
    df_kpi = pd.read_csv(RESULTADOS_PV_DIR / "pv_simulation_results.csv")
    df_sens_lcoe = pd.read_csv(RESULTADOS_PV_DIR / "sensibilidad_lcoe.csv")
    df_sens_npv = pd.read_csv(RESULTADOS_PV_DIR / "sensibilidad_npv.csv")

    if "Year" not in df_hourly.columns:
        df_hourly["Year"] = df_hourly["datetime"].dt.year
    
    localidades = df_hourly["Location"].unique()
    anios = sorted(df_hourly["Year"].unique()) # Sort years for consistent display
    return df_hourly, df_kpi, df_sens_lcoe, df_sens_npv, localidades, anios

df_hourly, df_kpi, df_sens_lcoe, df_sens_npv, localidades, anios = load_data()

# --- Helper Function (reused from notebook) ---
def crear_tornado(df, parametro_impacto_bajo, parametro_impacto_alto, titulo):
    df = df.copy()
    # Ensure columns exist before trying to calculate max_impact
    if parametro_impacto_bajo not in df.columns or parametro_impacto_alto not in df.columns:
        st.error(f"Missing one or both impact columns for tornado: {parametro_impacto_bajo}, {parametro_impacto_alto} in DataFrame for '{titulo}'")
        return go.Figure() # Return empty figure on error

    df['max_impact'] = df[[parametro_impacto_bajo, parametro_impacto_alto]].abs().max(axis=1)
    df = df.sort_values('max_impact', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['Parameter'],
        x=df[parametro_impacto_bajo],
        orientation='h',
        name='Impacto bajo',
        marker_color='steelblue'
    ))
    fig.add_trace(go.Bar(
        y=df['Parameter'],
        x=df[parametro_impacto_alto],
        orientation='h',
        name='Impacto alto',
        marker_color='indianred'
    ))
    fig.update_layout(
        barmode='overlay', # Changed from 'overlay' to 'group' or 'relative' if bars should not overlap, or keep 'overlay' if intended
        title=titulo,
        xaxis_title='Impacto (%)',
        yaxis_title='Parámetro',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- Streamlit App Layout ---
# st.set_page_config(layout="wide") # Use wide layout for better dashboard feel Commented out or removed
st.title("Dashboard Solar Integrado")

# --- Sidebar for Filters ---
st.sidebar.header("Filtros")
selected_localidad = st.sidebar.selectbox(
    "Selecciona país/localidad:",
    options=localidades,
    index=0 # Default to the first location
)

selected_anio = st.sidebar.selectbox(
    "Selecciona año:",
    options=anios,
    index=0 # Default to the first year
)

# --- Main Panel with Tabs for Organization ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Visión General", 
    "Datos TMY", 
    "Resultados Simulación PV", 
    "Análisis de Sensibilidad"
])

# --- Data Filtering based on selections ---
df_sel_hourly = df_hourly[(df_hourly["Location"] == selected_localidad) & (df_hourly["Year"] == selected_anio)].copy()
df_sel_kpi = df_kpi[df_kpi["Location"] == selected_localidad]
df_sel_sens_lcoe = df_sens_lcoe[df_sens_lcoe["Location"] == selected_localidad]
df_sel_sens_npv = df_sens_npv[df_sens_npv["Location"] == selected_localidad]

# --- Tab 1: Visión General (Potencia Horaria y KPIs) ---
with tab1:
    st.header(f"Visión General - {selected_localidad} ({selected_anio})")
    
    st.subheader("Curva horaria de potencia (AC Power)")
    if not df_sel_hourly.empty:
        fig_pot = px.line(df_sel_hourly, x="datetime", y="AC Power (kW)", title=f"Potencia horaria - {selected_localidad} {selected_anio}")
        st.plotly_chart(fig_pot, use_container_width=True)
    else:
        st.warning("No hay datos de potencia horaria para la selección actual.")

    st.subheader("KPIs Clave")
    if not df_sel_hourly.empty and not df_sel_kpi.empty:
        df_sel_hourly["date"] = df_sel_hourly["datetime"].dt.date
        energia_diaria = df_sel_hourly.groupby("date")["AC Power (kW)"].sum()
        energia_prom = energia_diaria.mean()
        # energia_total = energia_diaria.sum() # Not used in original KPIs display
        
        lcoe_val = df_sel_kpi["LCOE ($/kWh)"].values[0] if not df_sel_kpi.empty else "N/A"
        potencia_nominal = df_sel_hourly["AC Power (kW)"].max() if not df_sel_hourly.empty else 0
        
        cf_prom = 0
        if potencia_nominal > 0: # Avoid division by zero
            cf_diario = energia_diaria / (potencia_nominal * 24)
            cf_prom = cf_diario.mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Energía diaria promedio", value=f"{energia_prom:.2f} kWh/día" if energia_prom else "N/A")
        with col2:
            st.metric(label="LCOE", value=f"{lcoe_val:.4f} $/kWh" if isinstance(lcoe_val, (int, float)) else lcoe_val)
        with col3:
            st.metric(label="Factor de Capacidad promedio", value=f"{cf_prom*100:.2f} %" if cf_prom else "N/A")
    else:
        st.warning("No se pueden calcular KPIs para la selección actual.")


# --- Tab 2: Datos TMY ---
with tab2:
    st.header(f"Datos TMY Limpios - {selected_localidad}")
    tmy_file_path = DATOS_LIMPIOS_DIR / f"{selected_localidad.lower()}_TMY_final.csv"
    
    if tmy_file_path.exists():
        df_tmy_display = pd.read_csv(tmy_file_path, skiprows=2)
        st.subheader("Primeras 20 filas de datos TMY")
        st.dataframe(df_tmy_display.head(20), use_container_width=True)
        
        st.subheader("Gráficos de radiación horaria TMY")
        fig_tmy = px.line(df_tmy_display, x=df_tmy_display.index, y=["GHI", "DNI", "DHI"], 
                          labels={"value": "W/m²", "variable": "Componente", "index": "Hora del año"}, 
                          title=f"Radiación horaria TMY - {selected_localidad}")
        st.plotly_chart(fig_tmy, use_container_width=True)
    else:
        st.error(f"No se encontró el archivo TMY: {tmy_file_path}")

# --- Tab 3: Resultados Simulación PV ---
with tab3:
    st.header(f"Resultados de Simulación PV - {selected_localidad}")
    if not df_sel_kpi.empty:
        st.dataframe(df_sel_kpi, use_container_width=True)
    else:
        st.warning("No hay resultados de simulación PV para la localidad seleccionada.")

# --- Tab 4: Análisis de Sensibilidad ---
with tab4:
    st.header(f"Análisis de Sensibilidad - {selected_localidad}")

    st.subheader("Sensibilidad LCOE")
    if not df_sel_sens_lcoe.empty:
        fig_lcoe_tornado = crear_tornado(df_sel_sens_lcoe, "Impact Low (%)", "Impact High (%)", f"Tornado LCOE - {selected_localidad}")
        st.plotly_chart(fig_lcoe_tornado, use_container_width=True)
        st.dataframe(df_sel_sens_lcoe, use_container_width=True)
    else:
        st.warning(f"No hay datos de sensibilidad LCOE para {selected_localidad}.")

    st.subheader("Sensibilidad NPV")
    if not df_sel_sens_npv.empty:
        fig_npv_tornado = crear_tornado(df_sel_sens_npv, "Impact Low (%)", "Impact High (%)", f"Tornado NPV - {selected_localidad}")
        st.plotly_chart(fig_npv_tornado, use_container_width=True)
        st.dataframe(df_sel_sens_npv, use_container_width=True)
    else:
        st.warning(f"No hay datos de sensibilidad NPV para {selected_localidad}.")

# To run the app:
# 1. Save this code as streamlit_app.py
# 2. Open your terminal in the project directory (where streamlit_app.py and the data folders are)
# 3. Make sure your virtual environment is activated: source env/bin/activate
# 4. Install streamlit if you haven't: env/bin/python -m pip install streamlit
# 5. Run: streamlit run streamlit_app.py 