import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import pickle

# Configuración de la página
st.set_page_config(page_title="PoC Expansión B2B", layout="wide")

st.title("📊 Panel de Expansión Territorial - Grupo Data")
st.markdown("Prueba de Concepto (PoC) diseñada por Óscar Navarro para la optimización en la captación de Direcciones Asociadas.")

# Dividir la pantalla en dos pestañas
tab1, tab2 = st.tabs(["📍 Módulo 1: Location Intelligence", "🎯 Módulo 2: Lead Scoring Predictivo"])

with tab1:
    st.header("Identificación de Territorios Prioritarios")
    st.write("Mapa de calor simulado basado en densidad de pymes y penetración tecnológica.")
    
    # Crear un mapa centrado en España (Andalucía)
    mapa = folium.Map(location=[37.8881, -4.7793], zoom_start=7)
    
    # Añadir marcadores de oportunidad de negocio (Datos simulados)
    ubicaciones = [
        {"loc": [36.7213, -4.4214], "ciudad": "Málaga (Hub Tech)", "score": 95},
        {"loc": [37.3828, -5.9731], "ciudad": "Sevilla (Corporativo)", "score": 88},
        {"loc": [37.7692, -3.7902], "ciudad": "Jaén (Oportunidad Pyme)", "score": 75}
    ]
    
    for u in ubicaciones:
        folium.CircleMarker(
            location=u["loc"],
            radius=u["score"]/5,
            popup=f"{u['ciudad']} - Score: {u['score']}",
            color="blue",
            fill=True,
            fill_color="blue"
        ).add_to(mapa)
        
    st_folium(mapa, width=800, height=400)

with tab2:
    st.header("Calculadora de Idoneidad del Director Asociado")
    st.write("Introduce los datos del candidato para predecir su probabilidad de éxito en la red.")
    
    # Formularios de entrada para el usuario
    col1, col2 = st.columns(2)
    with col1:
        exp = st.slider("Años de Experiencia Directiva", 1, 20, 5)
        capital = st.number_input("Capital Disponible (en miles de €)", min_value=10, max_value=100, value=30)
    with col2:
        score_comercial = st.slider("Score Habilidades Comerciales (1-10)", 1, 10, 7)
        sector_retail = st.selectbox("¿Procede del sector Retail/Franquicias?", ["Sí", "No"])
        sector_val = 1 if sector_retail == "Sí" else 0

    if st.button("Analizar Candidato"):
        # Cargar el modelo guardado
        with open('modelo_scoring.pkl', 'rb') as archivo:
            modelo_cargado = pickle.load(archivo)
        
        # Preparar los datos del candidato actual
        datos_candidato = pd.DataFrame({
            'experiencia_directiva_anios': [exp],
            'capital_disponible_k': [capital],
            'score_habilidades_comerciales': [score_comercial],
            'sector_retail': [sector_val]
        })
        
        # Realizar la predicción (probabilidad)
        probabilidad = modelo_cargado.predict_proba(datos_candidato)[0][1] * 100
        
        # Mostrar el resultado
        st.subheader("Resultado de la Predicción:")
        if probabilidad >= 70:
            st.success(f"✅ PERFIL ÓPTIMO. Probabilidad de éxito: {probabilidad:.1f}%")
        elif probabilidad >= 40:
            st.warning(f"⚠️ PERFIL EN RIESGO. Probabilidad de éxito: {probabilidad:.1f}%")
        else:
            st.error(f"❌ PERFIL DESCARTADO. Probabilidad de éxito: {probabilidad:.1f}%")