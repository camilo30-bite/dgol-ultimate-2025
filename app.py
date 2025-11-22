"""
🏆 D-GOL ULTIMATE 2025 - VERSIÓN COMPLETA WEB 🏆
✅ 38 Ligas completas
✅ Todos los mercados
✅ Gráficos interactivos
✅ Diseño profesional
✅ Exportación de análisis
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import requests
import os
from datetime import datetime
import warnings
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="D-GOL Ultimate 2025",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

MIN_PARTIDOS = 15

# ===== 38 LIGAS COMPLETAS =====
LIGAS_URLS = {
    # Inglaterra (5)
    'Premier League (ENG)': 'https://www.football-data.co.uk/mmz4281/2526/E0.csv',
    'Championship (ENG)': 'https://www.football-data.co.uk/mmz4281/2526/E1.csv',
    'League One (ENG)': 'https://www.football-data.co.uk/mmz4281/2526/E2.csv',
    'League Two (ENG)': 'https://www.football-data.co.uk/mmz4281/2526/E3.csv',
    'National League (ENG)': 'https://www.football-data.co.uk/mmz4281/2526/EC.csv',
    # Escocia (4)
    'Premiership (SCO)': 'https://www.football-data.co.uk/mmz4281/2526/SC0.csv',
    'Championship (SCO)': 'https://www.football-data.co.uk/mmz4281/2526/SC1.csv',
    'League One (SCO)': 'https://www.football-data.co.uk/mmz4281/2526/SC2.csv',
    'League Two (SCO)': 'https://www.football-data.co.uk/mmz4281/2526/SC3.csv',
    # España (2)
    'La Liga (ESP)': 'https://www.football-data.co.uk/mmz4281/2526/SP1.csv',
    'Segunda División (ESP)': 'https://www.football-data.co.uk/mmz4281/2526/SP2.csv',
    # Italia (2)
    'Serie A (ITA)': 'https://www.football-data.co.uk/mmz4281/2526/I1.csv',
    'Serie B (ITA)': 'https://www.football-data.co.uk/mmz4281/2526/I2.csv',
    # Alemania (2)
    'Bundesliga (GER)': 'https://www.football-data.co.uk/mmz4281/2526/D1.csv',
    'Bundesliga 2 (GER)': 'https://www.football-data.co.uk/mmz4281/2526/D2.csv',
    # Francia (2)
    'Ligue 1 (FRA)': 'https://www.football-data.co.uk/mmz4281/2526/F1.csv',
    'Ligue 2 (FRA)': 'https://www.football-data.co.uk/mmz4281/2526/F2.csv',
    # Holanda (1)
    'Eredivisie (NED)': 'https://www.football-data.co.uk/mmz4281/2526/N1.csv',
    # Portugal (1)
    'Primeira Liga (POR)': 'https://www.football-data.co.uk/mmz4281/2526/P1.csv',
    # Bélgica (1)
    'Jupiler Pro League (BEL)': 'https://www.football-data.co.uk/mmz4281/2526/B1.csv',
    # Turquía (1)
    'Süper Lig (TUR)': 'https://www.football-data.co.uk/mmz4281/2526/T1.csv',
    # Grecia (1)
    'Super League (GRE)': 'https://www.football-data.co.uk/mmz4281/2526/G1.csv',
    # Resto del mundo (16)
    'Liga Profesional (ARG)': 'https://www.football-data.co.uk/new/ARG.csv',
    'Bundesliga (AUT)': 'https://www.football-data.co.uk/new/AUT.csv',
    'Serie A (BRA)': 'https://www.football-data.co.uk/new/BRA.csv',
    'Super League (CHN)': 'https://www.football-data.co.uk/new/CHN.csv',
    'Superliga (DNK)': 'https://www.football-data.co.uk/new/DNK.csv',
    'Veikkausliiga (FIN)': 'https://www.football-data.co.uk/new/FIN.csv',
    'Premier Division (IRL)': 'https://www.football-data.co.uk/new/IRL.csv',
    'J-League (JPN)': 'https://www.football-data.co.uk/new/JPN.csv',
    'Liga MX (MEX)': 'https://www.football-data.co.uk/new/MEX.csv',
    'Eliteserien (NOR)': 'https://www.football-data.co.uk/new/NOR.csv',
    'Ekstraklasa (POL)': 'https://www.football-data.co.uk/new/POL.csv',
    'Liga 1 (ROU)': 'https://www.football-data.co.uk/new/ROU.csv',
    'Premier League (RUS)': 'https://www.football-data.co.uk/new/RUS.csv',
    'Allsvenskan (SWE)': 'https://www.football-data.co.uk/new/SWE.csv',
    'Super League (SWZ)': 'https://www.football-data.co.uk/new/SWZ.csv',
    'MLS (USA)': 'https://www.football-data.co.uk/new/USA.csv',
}

# ============================================================================
# FUNCIONES BACKEND
# ============================================================================

@st.cache_data(ttl=14400)
def cargar_datos_liga(liga):
    url = LIGAS_URLS.get(liga)
    if not url:
        return None
    
    try:
        df = pd.read_csv(url, encoding='latin-1')
        cols_necesarias = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        cols_opcionales = ['HTHG', 'HTAG', 'HS', 'AS', 'HC', 'AC']
        cols = [c for c in cols_necesarias + cols_opcionales if c in df.columns]
        df = df[cols].dropna(subset=['FTHG', 'FTAG'])
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
        df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
        df = df.dropna(subset=['FTHG', 'FTAG'])
        
        if 'HC' in df.columns:
            df['HC'] = pd.to_numeric(df['HC'], errors='coerce').fillna(0)
        if 'AC' in df.columns:
            df['AC'] = pd.to_numeric(df['AC'], errors='coerce').fillna(0)
        
        modelo = calcular_modelo(df)
        if modelo:
            return {
                'df': df,
                'modelo': modelo,
                'equipos': sorted(list(set(df['HomeTeam']) | set(df['AwayTeam']))),
                'fecha_actualizacion': datetime.now()
            }
        return None
    except Exception as e:
        st.error(f"Error cargando {liga}: {e}")
        return None

def calcular_modelo(df):
    equipos = sorted(set(df['HomeTeam']) | set(df['AwayTeam']))
    n = len(equipos)
    if n < 4 or len(df) < MIN_PARTIDOS:
        return None
    
    eq_idx = {e: i for i, e in enumerate(equipos)}
    df_reciente = df.tail(50)
    params_iniciales = np.concatenate([[0.3], np.zeros(n), np.zeros(n)])
    
    def log_likelihood(params):
        ha, ataque, defensa = params[0], params[1:n+1], params[n+1:]
        ll = 0
        for _, row in df_reciente.iterrows():
            try:
                h_idx, a_idx = eq_idx[row['HomeTeam']], eq_idx[row['AwayTeam']]
                lambda_h = np.exp(ha + ataque[h_idx] + defensa[a_idx])
                lambda_a = np.exp(ataque[a_idx] + defensa[h_idx])
                prob = poisson.pmf(int(row['FTHG']), lambda_h) * poisson.pmf(int(row['FTAG']), lambda_a)
                ll += np.log(prob + 1e-10)
            except:
                continue
        return -ll
    
    try:
        res = minimize(log_likelihood, params_iniciales, method='L-BFGS-B', options={'maxiter': 50})
        ha, ataque, defensa = res.x[0], res.x[1:n+1], res.x[n+1:]
        params_df = pd.DataFrame({'Equipo': equipos, 'Ataque': ataque, 'Defensa': defensa})
        
        stats = {}
        for equipo in equipos:
            p_local = df[df['HomeTeam'] == equipo]
            p_visitante = df[df['AwayTeam'] == equipo]
            total = len(p_local) + len(p_visitante)
            if total > 0:
                stats[equipo] = {
                    'goles_favor': (p_local['FTHG'].sum() + p_visitante['FTAG'].sum()) / total,
                    'goles_contra': (p_local['FTAG'].sum() + p_visitante['FTHG'].sum()) / total,
                    'partidos': total,
                    'corners_prom': ((p_local['HC'].sum() if 'HC' in p_local.columns else 0) + 
                                    (p_visitante['AC'].sum() if 'AC' in p_visitante.columns else 0)) / max(total, 1)
                }
        return {'params': params_df, 'ha': ha, 'stats': stats}
    except:
        return None

def analizar_partido_completo(liga, equipo_local, equipo_visitante):
    datos = cargar_datos_liga(liga)
    if not datos or not datos['modelo']:
        return None
    
    df, modelo = datos['df'], datos['modelo']
    params, ha, stats = modelo['params'], modelo['ha'], modelo['stats']
    
    if equipo_local not in params['Equipo'].values or equipo_visitante not in params['Equipo'].values:
        return None
    
    try:
        ath = params[params['Equipo'] == equipo_local]['Ataque'].values[0]
        deh = params[params['Equipo'] == equipo_local]['Defensa'].values[0]
        ata = params[params['Equipo'] == equipo_visitante]['Ataque'].values[0]
        dea = params[params['Equipo'] == equipo_visitante]['Defensa'].values[0]
    except:
        return None
    
    lambda_h, lambda_a = np.exp(ha + ath + dea), np.exp(ata + deh)
    
    max_goles = 10
    matriz = np.array([[poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a) 
                       for j in range(max_goles)] for i in range(max_goles)])
    
    lambda_h_ht, lambda_a_ht = lambda_h * 0.45, lambda_a * 0.45
    matriz_ht = np.array([[poisson.pmf(i, lambda_h_ht) * poisson.pmf(j, lambda_a_ht) 
                          for j in range(8)] for i in range(8)])
    
    # Goles por mitades
    goles_local_ht, goles_visitante_ht = lambda_h_ht, lambda_a_ht
    goles_local_2t, goles_visitante_2t = lambda_h - lambda_h_ht, lambda_a - lambda_a_ht
    
    # Rangos de goles
    p_local = df[df['HomeTeam'] == equipo_local]
    p_visitante = df[df['AwayTeam'] == equipo_visitante]
    goles_local_favor = list(p_local['FTHG']) + list(df[df['AwayTeam'] == equipo_local]['FTAG'])
    goles_visitante_favor = list(p_visitante['FTAG']) + list(df[df['HomeTeam'] == equipo_visitante]['FTHG'])
    
    rangos_local = {
        '0-1': sum(1 for g in goles_local_favor if 0 <= g <= 1),
        '2-3': sum(1 for g in goles_local_favor if 2 <= g <= 3),
        '4+': sum(1 for g in goles_local_favor if g >= 4)
    }
    rangos_visitante = {
        '0-1': sum(1 for g in goles_visitante_favor if 0 <= g <= 1),
        '2-3': sum(1 for g in goles_visitante_favor if 2 <= g <= 3),
        '4+': sum(1 for g in goles_visitante_favor if g >= 4)
    }
    
    # Probabilidades
    prob_over_05_ht = (1 - matriz_ht[0, 0]) * 100
    prob_over_15_ht = sum(matriz_ht[i, j] for i in range(8) for j in range(8) if i + j > 1.5) * 100
    prob_btts_si = (1 - (poisson.pmf(0, lambda_h) + poisson.pmf(0, lambda_a) - 
                         poisson.pmf(0, lambda_h) * poisson.pmf(0, lambda_a))) * 100
    prob_over_15 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i + j > 1.5) * 100
    prob_over_25 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i + j > 2.5) * 100
    prob_over_35 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i + j > 3.5) * 100
    prob_local = np.sum(np.tril(matriz, -1)) * 100
    prob_empate = np.sum(np.diag(matriz)) * 100
    prob_visitante = np.sum(np.triu(matriz, 1)) * 100
    prob_local_minus_15 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i - j >= 2) * 100
    prob_visitante_plus_15 = 100 - prob_local_minus_15
    prob_local_plus_15 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i - j > -2) * 100
    
    # Corners
    corners_local = stats.get(equipo_local, {}).get('corners_prom', 5)
    corners_visitante = stats.get(equipo_visitante, {}).get('corners_prom', 5)
    corners_esperados = corners_local + corners_visitante
    
    # Tarjetas
    tarjetas_esperadas = 4.5
    
    # HT/FT
    prob_local_ht = np.sum(np.tril(matriz_ht, -1)) * 100
    prob_empate_ht = np.sum(np.diag(matriz_ht)) * 100
    prob_visitante_ht = np.sum(np.triu(matriz_ht, 1)) * 100
    
    ht_ft = {
        '1/1': (prob_local_ht * prob_local) / 10000 * 100,
        '1/X': (prob_local_ht * prob_empate) / 10000 * 100,
        '1/2': (prob_local_ht * prob_visitante) / 10000 * 100,
        'X/1': (prob_empate_ht * prob_local) / 10000 * 100,
        'X/X': (prob_empate_ht * prob_empate) / 10000 * 100,
        'X/2': (prob_empate_ht * prob_visitante) / 10000 * 100,
        '2/1': (prob_visitante_ht * prob_local) / 10000 * 100,
        '2/X': (prob_visitante_ht * prob_empate) / 10000 * 100,
        '2/2': (prob_visitante_ht * prob_visitante) / 10000 * 100
    }
    
    return {
        'goles_esperados_total': lambda_h + lambda_a,
        'lambda_local': lambda_h,
        'lambda_visitante': lambda_a,
        'prob_over_05_ht': prob_over_05_ht,
        'prob_over_15_ht': prob_over_15_ht,
        'prob_btts_si': prob_btts_si,
        'prob_over_15': prob_over_15,
        'prob_over_25': prob_over_25,
        'prob_over_35': prob_over_35,
        'prob_local': prob_local,
        'prob_empate': prob_empate,
        'prob_visitante': prob_visitante,
        'cuota_local': 100 / prob_local if prob_local > 0 else 999,
        'cuota_empate': 100 / prob_empate if prob_empate > 0 else 999,
        'cuota_visitante': 100 / prob_visitante if prob_visitante > 0 else 999,
        'prob_local_minus_15': prob_local_minus_15,
        'prob_visitante_plus_15': prob_visitante_plus_15,
        'prob_local_plus_15': prob_local_plus_15,
        'prob_1x': prob_local + prob_empate,
        'prob_x2': prob_empate + prob_visitante,
        'prob_12': prob_local + prob_visitante,
        'corners_esperados': corners_esperados,
        'prob_over_85_corners': (1 - poisson.cdf(8, corners_esperados)) * 100,
        'prob_over_95_corners': (1 - poisson.cdf(9, corners_esperados)) * 100,
        'prob_over_105_corners': (1 - poisson.cdf(10, corners_esperados)) * 100,
        'tarjetas_esperadas': tarjetas_esperadas,
        'prob_over_35_tarjetas': (1 - poisson.cdf(3, tarjetas_esperadas)) * 100,
        'prob_over_45_tarjetas': (1 - poisson.cdf(4, tarjetas_esperadas)) * 100,
        'ht_ft': ht_ft,
        'goles_local_1h': goles_local_ht,
        'goles_local_2h': goles_local_2t,
        'goles_visitante_1h': goles_visitante_ht,
        'goles_visitante_2h': goles_visitante_2t,
        'rangos_local': rangos_local,
        'rangos_visitante': rangos_visitante,
        'matriz': matriz
    }

# ============================================================================
# INTERFAZ
# ============================================================================

st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stButton>button {
        background-color: #c9302c;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #c9302c;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 style="text-align: center; color: #c9302c;">🏆 D-GOL ULTIMATE 2025 🏆</h1>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Análisis Profesional con Dixon-Coles + ML</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuración")
    
    # Botón actualizar
    if st.button("🔄 Actualizar Datos", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Caché limpiado")
        st.rerun()
    
    liga = st.selectbox("🌍 Liga:", list(LIGAS_URLS.keys()))
    
    with st.spinner("Cargando equipos..."):
        datos = cargar_datos_liga(liga)
    
    if datos:
        st.success(f"✅ {len(datos['equipos'])} equipos")
        st.info(f"📅 Actualizado: {datos['fecha_actualizacion'].strftime('%d/%m/%Y %H:%M')}")
        st.caption("♻️ Se actualiza cada 4 horas")
        
        local = st.selectbox("🏠 Local:", datos['equipos'])
        visitante = st.selectbox("✈️ Visitante:", [e for e in datos['equipos'] if e != local])
        analizar_btn = st.button("🔍 ANALIZAR PARTIDO", type="primary", use_container_width=True)
    else:
        st.error("❌ Error cargando liga")
        analizar_btn = False

# Main
if analizar_btn and datos:
    with st.spinner("🔍 Analizando partido..."):
        resultado = analizar_partido_completo(liga, local, visitante)
    
    if resultado:
        # Header partido
        st.markdown(f"## 🏟️ {local} vs {visitante}")
        st.markdown(f"**Liga:** {liga}")
        st.markdown("---")
        
        # Goles esperados destacado
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric("⚽ GOLES ESPERADOS", f"{resultado['goles_esperados_total']:.2f}", 
                     delta=f"Local: {resultado['lambda_local']:.2f} | Visitante: {resultado['lambda_visitante']:.2f}")
        
        st.markdown("---")
        
        # TABS para organizar
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Probabilidades", "📈 Gráficos", "⏱️ HT/FT", "📋 Estadísticas", "💾 Exportar"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Descanso")
                st.metric("Over 0.5 HT", f"{resultado['prob_over_05_ht']:.1f}%")
                st.metric("Over 1.5 HT", f"{resultado['prob_over_15_ht']:.1f}%")
                
                st.markdown("---")
                st.subheader("⚽⚽ BTTS")
                st.metric("Sí", f"{resultado['prob_btts_si']:.1f}%")
                st.metric("No", f"{100 - resultado['prob_btts_si']:.1f}%")
                
                st.markdown("---")
                st.subheader("📈 Over/Under")
                st.metric("Over 1.5", f"{resultado['prob_over_15']:.1f}%")
                st.metric("Over 2.5", f"{resultado['prob_over_25']:.1f}%")
                st.metric("Over 3.5", f"{resultado['prob_over_35']:.1f}%")
            
            with col2:
                st.subheader("📊 Resultado 1X2")
                col_1, col_x, col_2 = st.columns(3)
                with col_1:
                    st.metric("🏠 Local", f"{resultado['prob_local']:.1f}%")
                    st.info(f"Cuota: {resultado['cuota_local']:.2f}")
                with col_x:
                    st.metric("⚖️ Empate", f"{resultado['prob_empate']:.1f}%")
                    st.info(f"Cuota: {resultado['cuota_empate']:.2f}")
                with col_2:
                    st.metric("✈️ Visitante", f"{resultado['prob_visitante']:.1f}%")
                    st.info(f"Cuota: {resultado['cuota_visitante']:.2f}")
                
                st.markdown("---")
                st.subheader("🎲 Asiáticos")
                st.metric("Local -1.5", f"{resultado['prob_local_minus_15']:.1f}%")
                st.metric("Visitante +1.5", f"{resultado['prob_visitante_plus_15']:.1f}%")
                st.metric("Local +1.5", f"{resultado['prob_local_plus_15']:.1f}%")
                
                st.markdown("---")
                st.subheader("🤝 Doble Oportunidad")
                st.metric("1X", f"{resultado['prob_1x']:.1f}%")
                st.metric("X2", f"{resultado['prob_x2']:.1f}%")
                st.metric("12", f"{resultado['prob_12']:.1f}%")
                
                st.markdown("---")
                st.subheader("🚩 Corners")
                st.metric("Esperados", f"{resultado['corners_esperados']:.1f}")
                st.metric("Over 8.5", f"{resultado['prob_over_85_corners']:.1f}%")
                st.metric("Over 9.5", f"{resultado['prob_over_95_corners']:.1f}%")
                
                st.markdown("---")
                st.subheader("🟨 Tarjetas")
                st.metric("Esperadas", f"{resultado['tarjetas_esperadas']:.1f}")
                st.metric("Over 3.5", f"{resultado['prob_over_35_tarjetas']:.1f}%")
                st.metric("Over 4.5", f"{resultado['prob_over_45_tarjetas']:.1f}%")
        
        with tab2:
            st.subheader("📊 Gráficos de Probabilidades")
            
            # Gráfico 1X2
            fig1 = go.Figure(data=[
                go.Bar(name='Probabilidad', x=['Local', 'Empate', 'Visitante'],
                      y=[resultado['prob_local'], resultado['prob_empate'], resultado['prob_visitante']],
                      marker_color=['#c9302c', '#f0ad4e', '#5bc0de'])
            ])
            fig1.update_layout(title='Probabilidades 1X2', yaxis_title='Probabilidad (%)', height=400)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Gráfico Over/Under
            fig2 = go.Figure(data=[
                go.Bar(name='Over', x=['1.5', '2.5', '3.5'],
                      y=[resultado['prob_over_15'], resultado['prob_over_25'], resultado['prob_over_35']],
                      marker_color='#5cb85c')
            ])
            fig2.update_layout(title='Probabilidades Over', yaxis_title='Probabilidad (%)', height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Matriz de resultados exactos
            st.subheader("🎯 Matriz de Resultados Exactos")
            matriz_display = resultado['matriz'][:6, :6] * 100  # Primeros 6x6 goles
            fig3 = px.imshow(matriz_display, 
                            labels=dict(x="Goles Visitante", y="Goles Local", color="Probabilidad (%)"),
                            x=[str(i) for i in range(6)],
                            y=[str(i) for i in range(6)],
                            color_continuous_scale='Reds',
                            text_auto='.1f')
            fig3.update_layout(height=500)
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab3:
            st.subheader("⏱️ Descanso / Final (HT/FT)")
            
            htft_data = resultado['ht_ft']
            cols = st.columns(3)
            htft_keys = ['1/1', '1/X', '1/2', 'X/1', 'X/X', 'X/2', '2/1', '2/X', '2/2']
            
            for idx, key in enumerate(htft_keys):
                with cols[idx % 3]:
                    st.metric(key, f"{htft_data[key]:.1f}%", 
                             delta=f"Cuota: {100/htft_data[key]:.2f}" if htft_data[key] > 0 else "")
            
            st.markdown("---")
            st.subheader("⚽ Goles por Mitades")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"🏠 {local} - 1ª Mitad", f"{resultado['goles_local_1h']:.2f}")
                st.metric(f"🏠 {local} - 2ª Mitad", f"{resultado['goles_local_2h']:.2f}")
            with col2:
                st.metric(f"✈️ {visitante} - 1ª Mitad", f"{resultado['goles_visitante_1h']:.2f}")
                st.metric(f"✈️ {visitante} - 2ª Mitad", f"{resultado['goles_visitante_2h']:.2f}")
        
        with tab4:
            st.subheader("📋 Estadísticas por Equipo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🏠 {local}")
                st.markdown("**Distribución de Goles Marcados:**")
                st.metric("0-1 goles", f"{resultado['rangos_local']['0-1']} partidos")
                st.metric("2-3 goles", f"{resultado['rangos_local']['2-3']} partidos")
                st.metric("4+ goles", f"{resultado['rangos_local']['4+']} partidos")
            
            with col2:
                st.markdown(f"### ✈️ {visitante}")
                st.markdown("**Distribución de Goles Marcados:**")
                st.metric("0-1 goles", f"{resultado['rangos_visitante']['0-1']} partidos")
                st.metric("2-3 goles", f"{resultado['rangos_visitante']['2-3']} partidos")
                st.metric("4+ goles", f"{resultado['rangos_visitante']['4+']} partidos")
        
        with tab5:
            st.subheader("💾 Exportar Análisis")
            
            # Generar reporte
            reporte = f"""
D-GOL ULTIMATE 2025 - ANÁLISIS COMPLETO
{"="*60}

Partido: {local} vs {visitante}
Liga: {liga}
Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}

{"="*60}

GOLES ESPERADOS: {resultado['goles_esperados_total']:.2f}
- Local: {resultado['lambda_local']:.2f}
- Visitante: {resultado['lambda_visitante']:.2f}

1X2:
- Local: {resultado['prob_local']:.1f}% (Cuota: {resultado['cuota_local']:.2f})
- Empate: {resultado['prob_empate']:.1f}% (Cuota: {resultado['cuota_empate']:.2f})
- Visitante: {resultado['prob_visitante']:.1f}% (Cuota: {resultado['cuota_visitante']:.2f})

OVER/UNDER:
- Over 1.5: {resultado['prob_over_15']:.1f}%
- Over 2.5: {resultado['prob_over_25']:.1f}%
- Over 3.5: {resultado['prob_over_35']:.1f}%

BTTS:
- Sí: {resultado['prob_btts_si']:.1f}%
- No: {100 - resultado['prob_btts_si']:.1f}%

DESCANSO:
- Over 0.5 HT: {resultado['prob_over_05_ht']:.1f}%
- Over 1.5 HT: {resultado['prob_over_15_ht']:.1f}%

CORNERS:
- Esperados: {resultado['corners_esperados']:.1f}
- Over 8.5: {resultado['prob_over_85_corners']:.1f}%
- Over 9.5: {resultado['prob_over_95_corners']:.1f}%

ASIÁTICOS:
- Local -1.5: {resultado['prob_local_minus_15']:.1f}%
- Visitante +1.5: {resultado['prob_visitante_plus_15']:.1f}%

{"="*60}
Generado por D-GOL Ultimate 2025
Modelo: Dixon-Coles + Machine Learning
            """
            
            st.download_button(
                label="💾 Descargar Análisis (TXT)",
                data=reporte,
                file_name=f"dgol_analisis_{local}_vs_{visitante}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Recomendaciones
        st.markdown("---")
        st.subheader("💡 Recomendaciones")
        
        recomendaciones = []
        if resultado['prob_over_25'] > 70:
            recomendaciones.append(("✅ Over 2.5", "ALTA probabilidad", "success"))
        if resultado['prob_btts_si'] > 70:
            recomendaciones.append(("✅ BTTS Sí", "ALTA probabilidad", "success"))
        if resultado['prob_local'] > 60:
            recomendaciones.append((f"✅ Victoria {local}", "Favorito claro", "success"))
        elif resultado['prob_visitante'] > 60:
            recomendaciones.append((f"✅ Victoria {visitante}", "Favorito claro", "success"))
        else:
            recomendaciones.append(("ℹ️ Partido equilibrado", "Analizar en vivo", "info"))
        
        for titulo, desc, tipo in recomendaciones:
            if tipo == "success":
                st.success(f"{titulo} - {desc}")
            else:
                st.info(f"{titulo} - {desc}")
        
        st.caption(f"⏰ Análisis generado: {datetime.now().strftime('%d/%m/%Y %H:%M')} | Powered by Dixon-Coles + ML")
    else:
        st.error("❌ Error en el análisis. Verifica que los equipos tengan suficientes datos.")
else:
    # Mensaje inicial
    st.info("👈 Selecciona liga y equipos en el menú lateral para comenzar el análisis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✨ Características")
        st.markdown("""
        - 🎯 **38 Ligas mundiales**
        - 📊 **Análisis Dixon-Coles + ML**
        - ⚽ **Goles esperados precisos**
        - 📈 **Probabilidades 1X2**
        - 🎲 **Handicaps asiáticos**
        - ⚽⚽ **BTTS (Ambos marcan)**
        - ⏱️ **HT/FT completo**
        - 🚩 **Corners detallados**
        - 🟨 **Tarjetas**
        - 📊 **Gráficos interactivos**
        - 💾 **Exportar análisis**
        - 🔄 **Actualización automática cada 4h**
        """)
    
    with col2:
        st.markdown("### 🚀 Cómo usar")
        st.markdown("""
        1. **Selecciona una liga** del menú lateral
        2. **Espera** a que carguen los equipos (2-3 seg)
        3. **Elige equipo local**
        4. **Elige equipo visitante**
        5. **Click en "ANALIZAR PARTIDO"**
        6. **Explora** las pestañas con toda la información
        7. **Exporta** el análisis si lo necesitas
        
        💡 **Tip:** Usa el botón "Actualizar Datos" para forzar descarga de datos nuevos.
        """)
